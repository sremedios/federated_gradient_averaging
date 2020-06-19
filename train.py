import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import datetime
import time
import json
import pickle
import requests
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.cnn import *
from utils.forward import *
from utils.misc import *
from utils.load_mnist import *

def federate_grads(URL, client_grad, client_headers, sleep_delay=0.01):
    ########## SEND GRADIENT ##########
    put_successful = False
    while not put_successful:
        data = pickle.dumps(client_grad)
        response = requests.put(
            URL + "put_grad",
            data=data, 
            headers=client_headers,
        )
        put_successful = pickle.loads(response.content)
        
        time.sleep(sleep_delay)

    ########## GET AVG GRADIENT ##########
    server_grad = None
    while server_grad is None:
        
        # exit if another site converged
        response = requests.get(URL + "get_converged", headers=client_headers)
        others_converged = pickle.loads(response.content)
        if others_converged:
            print("\nAnother site has converged. Terminating.\n")
#             requests.put(
#                 URL + "terminate_request",
#                 headers=client_headers,
#             )
            sys.exit()

        # otherwise continue as normal
        try:
            response = requests.get(URL + "get_avg_grad", headers=client_headers)
        except requests.exceptions.ConnectionError:
            time.sleep(3)
        server_grad = pickle.loads(response.content)
        
        time.sleep(sleep_delay)

    return server_grad

if __name__ == '__main__':
     
    #################### HYPERPARAMS / ARGS ####################

    WEIGHT_DIR = Path(sys.argv[1])
    TB_LOG_DIR = Path(sys.argv[2])
    SITE = sys.argv[3].upper()
    MODE = sys.argv[4]
    GPU_ID = sys.argv[5]
    
    ### GPU settings ###
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    # cut memory consumption in half if not only local training
    if MODE != "local":
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)],
        )
        

    BATCH_SIZE = 2**12
    LEARNING_RATE = 1e-3
    epsilon = 1e-4
    best_val_loss = 100000 
    CONVERGENCE_EPOCH_LIMIT = 10
    convergence_epoch_counter = 0
    
    MODEL_NAME = "CNN_mode_{}_site_{}".format(MODE, SITE)
    experiment = "{}_lr_{}".format(MODEL_NAME, LEARNING_RATE)
    WEIGHT_DIR = WEIGHT_DIR / MODEL_NAME
    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    
    TB_LOG_DIR = TB_LOG_DIR / "{}_{}".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        experiment,    
    )
    
    train_summary_writer = tf.summary.create_file_writer(str(TB_LOG_DIR / "train"))
    val_summary_writer = tf.summary.create_file_writer(str(TB_LOG_DIR / "val"))
    
    
    
    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not d.exists():
            d.mkdir(parents=True)
            
    #################### SERVER SETUP ####################

    URL = "http://127.0.0.1:10203/"
    if MODE == "local":
        k_init = get_kernel_initializer()
    elif MODE == "federated":
        r = requests.get(URL + "kernel_init")
        k_init = pickle.loads(r.content)
    else:
        print("Invalid mode: {}".format(MODE))
        sys.exit()
    
    #################### CLIENT SETUP ####################

    client_headers = {
        "content-type": "binary tensor", 
        "site": SITE, 
    }

    #################### MODEL ####################

    model = cnn(k_init)
    
    model.save_weights(str(WEIGHT_DIR / "init_weights.h5"))
     
    with open(str(MODEL_PATH), 'w') as f:
        json.dump(model.to_json(), f)
    
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    # set up metrics
    train_acc = tf.keras.metrics.Accuracy(name="train_acc")
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_acc = tf.keras.metrics.Accuracy(name="val_acc")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    
    #################### LOAD DATA ####################
    print("Loading MNIST...")
    x, y, *_ = prepare_mnist(SITE)

    # if training on both sets, split data in half so the model doesn't 
    # train with "double data" compared to single site
    # also cut `BATCH_SIZE` in half
    if MODE == "federated":
        x = x[:len(x)//2]
        y = y[:len(y)//2]
        BATCH_SIZE = BATCH_SIZE // 2

    print("Dataset count: {}".format(len(x)))
    split = int(np.ceil(0.8 * len(x)))

    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    print("Train set count: {}\nValidation set count: {}\n"\
          .format(len(x_train), len(x_val)))
    
    
    #################### SETUP ####################
    print("\n{} TRAINING NETWORK {}\n".format(
        "#"*20,
        "#"*20,
    ))
    TEMPLATE = (
        "\r{: >12} Epoch {: >3d} | "
        "Epoch Step {: >4d}/{: >4d} | "
        "Global Step {: >6d} | "
        "{: >5.2f}s/step | "
        "ETA: {: >7.2f}s"
    )
    
    elapsed = 0.0
    cur_epoch = 0
    global_train_step = 0
    global_val_step = 0

    N_TRAIN_STEPS = len(x_train) // BATCH_SIZE
    N_VAL_STEPS = len(x_val) // BATCH_SIZE
    
    while(True):
        cur_epoch += 1
        epoch_st = time.time()

        # Reset metrics every epoch
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()

        #################### TRAINING ####################

        for cur_step, i in enumerate(range(0, len(x_train), BATCH_SIZE)):

            st = time.time()

            xs = x_train[i:i+BATCH_SIZE]
            ys = y_train[i:i+BATCH_SIZE]

            # Logits are the predictions here
            client_grads, loss, preds = forward(
                inputs=(xs, ys),
                model=model,
                loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits,
                training=True,
            )
            
            if MODE == "local":
                grads = client_grads
            elif MODE == "federated":
                grads = federate_grads(URL, client_grads, client_headers)
                
                    
            opt.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss)
            train_acc.update_state(ys, tf.argmax(preds, axis=1))


            #################### END-OF-STEP CALCULATIONS ####################
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=global_train_step)
                tf.summary.scalar('train_acc', train_acc.result(), step=global_train_step)
            global_train_step += 1

            en = time.time()
            elapsed = running_average(elapsed, en-st, cur_step+1)
            eta = (N_TRAIN_STEPS - cur_step) * elapsed
            print(
                TEMPLATE.format(
                        "Training",
                        cur_epoch,
                        cur_step,
                        N_TRAIN_STEPS,
                        global_train_step,
                        elapsed,
                        eta,
                ),
                end="",
            )

        print()

        #################### VALIDATION ####################

        for cur_step, i in enumerate(range(0, len(x_val), BATCH_SIZE)):

            st = time.time()

            xs = x_val[i:i+BATCH_SIZE]
            ys = y_val[i:i+BATCH_SIZE]

            # Logits are the predictions here
            loss, preds = forward(
                inputs=(xs, ys),
                model=model,
                loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits,
                training=False,
            )

            val_loss.update_state(loss)
            val_acc.update_state(ys, tf.argmax(preds, axis=1))

            #################### END-OF-STEP CALCULATIONS ####################
            
            with val_summary_writer.as_default():
                tf.summary.scalar('val_loss', val_loss.result(), step=global_val_step)
                tf.summary.scalar('val_acc', val_acc.result(), step=global_val_step)

            global_val_step += 1

            en = time.time()
            elapsed = running_average(elapsed, en-st, cur_step+1)
            eta = (N_VAL_STEPS - cur_step) * elapsed
            print(
                TEMPLATE.format(
                        "Validation",
                        cur_epoch,
                        cur_step,
                        N_VAL_STEPS,
                        global_val_step,
                        elapsed,
                        eta,
                ),
                end="",
            )

        #################### END-OF-EPOCH CALCULATIONS ####################
            
        epoch_en = time.time()
        print("\n\tEpoch elapsed time: {:.2f}s".format(epoch_en-epoch_st))
        
        # save weights
        improved_loss_cond = val_loss.result() < best_val_loss
        loss_diff_cond = np.abs(val_loss.result() - best_val_loss) > epsilon
        if improved_loss_cond and loss_diff_cond:
            print("\t\tLoss improved from {:.4f} to {:.4f}".format(best_val_loss, val_loss.result()))
            # update best
            best_val_loss = val_loss.result()
            # save weights
            model.save_weights(str(WEIGHT_DIR / "best_weights.h5"))
            # reset convergence counter
            convergence_epoch_counter = 0
        else:
            convergence_epoch_counter += 1 

        # check for exit 
        if convergence_epoch_counter >= CONVERGENCE_EPOCH_LIMIT:
            converged_success = False
            # Alert the server when done
            while not converged_success:
                try:
                    response = requests.put(
                        URL + "put_converged",
                        headers=client_headers,
                    )
                except requests.exceptions.ConnectionError:
                    time.sleep(3)
                else:
                    converged_success = pickle.loads(response.content)
                    time.sleep(1)
            
            print("\nConvergence criteria met. Terminating.\n")
#             requests.put(
#                 URL + "terminate_request",
#                 headers=client_headers,
#             )
            sys.exit()