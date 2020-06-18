import pickle
import requests

import matplotlib.pyplot as plt

import json
import numpy as np
import os
from pathlib import Path
from subprocess import Popen, PIPE
import sys
import time

import tensorflow as tf
import tensorflow.keras.backend as K

from utils.grad_ops import *
from utils import utils, patch_ops
from utils import preprocess

from models.losses import *
from models.unet import unet

sys.stdout.write("\n\033[0;0m")

def prepare_mnist(site):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    fst_half_idx = np.where(y_train < 5)[0]
    snd_half_idx = np.where(y_train >= 5)[0]
    thresh = min(len(fst_half_idx), len(snd_half_idx))

    if site=="ALL":
        train_idx = np.where(y_train <10)[0]
    elif site=="CNRM":
        train_idx = np.where(y_train < 5)[0]
    else:
        train_idx = np.where(y_train >= 5)[0]

    # balance counts
    train_idx = train_idx[:thresh]

    return (
        x_train[train_idx].astype(np.float32), 
        y_train[train_idx].astype(np.int32), 
        x_test.astype(np.float32), 
        y_test.astype(np.int32),
    )
    

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

def show_progbar(cur_epoch, total_epochs, cur_step, num_instances, loss, acc, color_code):
    TEMPLATE = "\r{}Epoch {}/{} [{:{}<{}}] Loss: {:>3.4f} Acc: {:>3.2%}"
    progbar_length = 20

    sys.stdout.write(TEMPLATE.format(
        color_code,
        cur_epoch,
        total_epochs,
        "=" * min(int(progbar_length*(cur_step/num_instances)), progbar_length),
        "-",
        progbar_length,
        loss,
        acc,
    ))

    sys.stdout.flush()

def step_batch_gradient(inputs, model):
    x, y = inputs

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        y_true = tf.Variable(y)
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
        )

    batch_grads = tape.gradient(batch_loss, model.trainable_variables)

    batch_preds = tf.nn.softmax(logits)

    return batch_grads, batch_loss, batch_preds

def step_federated_grad(client_grad):
    ########## SEND GRADIENT ##########
    r = False
    while not r:
        data = pickle.dumps(client_grad)
        r = requests.put(URL + "put_grad",
                         data=data, headers=headers)
        r = pickle.loads(r.content)
        time.sleep(1)

    ########## GET AVG GRADIENT ##########
    server_grad = False
    while not server_grad:
        try:
            r = requests.get(URL + "get_avg_grad", headers=headers)
        except requests.exceptions.ConnectionError:
            time.sleep(3)
        except KeyboardInterrupt as e:
            sys.stdout.write("\n\033[0;0mTerminated by {}.\n".format(e))
        server_grad = pickle.loads(r.content)
        time.sleep(1)

    return client_grad, server_grad

def step_batch_val(inputs, model):
    x, y = inputs

    logits = model(x, training=True)
    y_true = tf.Variable(y)

    batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=logits,
    )

    batch_preds = tf.nn.softmax(logits)

    return batch_loss, batch_preds

if __name__ == "__main__":

    try:

        ########## GPU SETUP ##########

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
        conf = tf.compat.v1.ConfigProto(gpu_options=opts)
        tf.compat.v1.enable_eager_execution(config=conf)

        ########## SERVER SETUP ##########

        URL = "http://127.0.0.1:10203/"

        ########## HYPERPARAMETER SETUP ##########

        num_channels = 1
        num_epochs = 50
        batch_size = 1024
        start_time = utils.now()
        num_classes = 10
        learning_rate = 1e-3
        train_color_code = "\033[0;32m"
        val_color_code = "\033[0;36m"
        CONVERGENCE_EPOCH_LIMIT = 10
        epsilon = 1e-4
        loc = sys.argv[1]
        try:
            with open("headers.json", "r") as fp:
                headers = json.load(fp)
        except:
            print("Institution header file not present; exiting.")
            sys.exit()

        headers['institute'] = loc

        ########## DIRECTORY SETUP ##########

        MODEL_NAME = "mnist"
        WEIGHT_DIR = Path("models/weights") / MODEL_NAME
        RESULTS_DIR = Path("results") / MODEL_NAME

        for d in [WEIGHT_DIR, RESULTS_DIR]:
            if not d.exists():
                d.mkdir(parents=Path('.'))

        MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
        HISTORY_PATH = WEIGHT_DIR / (MODEL_NAME + ".history.json")
        TRAIN_CURVE_FILENAME = RESULTS_DIR / "training_curve.csv"

        ########## WEIGHTS AND OPTIMIZER SETUP ##########

        # shared weights from server
        r = requests.get(URL + "init_layers")
        shared_weights = pickle.loads(r.content)


        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        #opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

        ######### MODEL AND CALLBACKS #########
        layer_iter = iter(shared_weights)
        inputs = tf.keras.layers.Input((None, None, num_channels))
        a = next(layer_iter)(inputs)
        a = tf.keras.layers.MaxPooling2D(2)(a)
        a = next(layer_iter)(a)
        a = tf.keras.layers.MaxPooling2D(2)(a)
        a = next(layer_iter)(a)
        a = tf.keras.layers.MaxPooling2D(2)(a)
        a = next(layer_iter)(a)
        a = tf.keras.layers.MaxPooling2D(2)(a)
        a = tf.keras.layers.GlobalAveragePooling2D()(a)

        a = next(layer_iter)(a)
        model = tf.keras.models.Model(inputs=inputs,outputs=a)


        shared_layer_indices = []
        shared_layer_names = [l.name for l in iter(shared_weights)]
        for i, l in enumerate(model.layers):
            if l.name in shared_layer_names:
                shared_layer_indices.append(i)

        INIT_WEIGHT_PATH = WEIGHT_DIR / "init_weights_{}.h5".format(loc)
        model.save_weights(str(INIT_WEIGHT_PATH))
        json_string = model.to_json()
        with open(str(MODEL_PATH), 'w') as f:
            json.dump(json_string, f)

        print(model.summary(line_length=75))


        ######### PREPROCESS TRAINING DATA #########

        x_train, y_train, x_val, y_val = prepare_mnist(loc)
        x_train = np.reshape(x_train, x_train.shape + (1,))
        x_val = np.reshape(x_val, x_val.shape + (1,))

        ######### TRAINING #########
        best_val_loss = 1e6
        best_val_acc = 0
        loss_diff = 1e6
        convergence_epoch_counter = 0
        print()
        best_epoch = 1

        train_accuracy = tf.keras.metrics.Accuracy(name="train_acc")
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        val_accuracy = tf.keras.metrics.Accuracy(name="val_acc")
        val_loss = tf.keras.metrics.Mean(name="val_loss")

        for cur_epoch in range(num_epochs):

            ### TRAIN STEP ###
            num_train_steps = int(len(x_train))
            print("\n{}Training...".format(train_color_code))
            for i in range(0, num_train_steps, batch_size):
                x = x_train[i: i + batch_size]
                y = y_train[i: i + batch_size]

                batch_grads, batch_loss, batch_preds = step_batch_gradient((x, y), model)

                client_grad, server_grad = step_federated_grad(batch_grads)
                opt.apply_gradients(zip(server_grad, model.trainable_variables))
                #opt.apply_gradients(zip(batch_grads, model.trainable_variables))

                train_accuracy.update_state(y, tf.argmax(batch_preds, axis=1))
                train_loss.update_state(batch_loss)

                show_progbar(
                    cur_epoch + 1,
                    num_epochs,
                    i + 1,
                    num_train_steps,
                    train_loss.result(),
                    train_accuracy.result(),
                    train_color_code,
                )



            ### VAL STEP ###
            num_val_steps = int(len(x_val))
            print("\n{}Validating...".format(val_color_code))
            for i in range(0, num_val_steps, batch_size):
                x = x_val[i: i + batch_size]
                y = y_val[i: i + batch_size]

                batch_loss, batch_preds = step_batch_val((x, y), model)

                val_accuracy.update_state(y, tf.argmax(batch_preds, axis=1))
                val_loss.update_state(batch_loss)

                show_progbar(
                    cur_epoch + 1,
                    num_epochs,
                    i + 1,
                    num_val_steps,
                    val_loss.result(),
                    val_accuracy.result(),
                    val_color_code,
                )


            ########## END OF EPOCH CALCS ##########

            with open(str(TRAIN_CURVE_FILENAME), 'a') as f:
                f.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    cur_epoch + 1,
                    train_loss.result(),
                    train_accuracy.result(),
                    val_loss.result(),
                    val_accuracy.result(),
                ))

            if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
                print("\nConvergence criteria reached.\nTerminating")
                break

            if val_loss.result() > best_val_loss and\
                np.abs(val_loss.result() - best_val_los) > epsilon:
                convergence_epoch_counter += 1
            else:
                convergence_epoch_counter = 0

            if val_loss.result() < best_val_loss:
                best_epoch = cur_epoch + 1
                best_val_loss = val_loss.result()
                best_val_acc = val_accuracy.result()
                model.save_weights(
                    str(WEIGHT_DIR / "{}_epoch_{:04d}_val_loss_{:.4f}_weights.hdf5".format(
                        loc,
                        cur_epoch,
                        val_loss.result(),
                    ))
                )

        sys.stdout.write("\n\033[0;0m")

    except Exception as e:
        sys.stdout.write("\n\033[0;0mTerminated by {}.\n".format(e))
