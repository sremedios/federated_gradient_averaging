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
from utils.opt_utils import *

def federate_vals(URL, client_val, client_headers, sleep_delay=0.01):
    ########## SEND ##########
    put_successful = False
    while not put_successful:
        data = pickle.dumps(client_val)
        response = requests.put(
            URL + "put_val",
            data=data, 
            headers=client_headers,
        )
        put_successful = pickle.loads(response.content)
        
        time.sleep(sleep_delay)

    ########## GET AVG ##########
    server_val = None
    while server_val is None:
        

        # otherwise continue as normal
        try:
            response = requests.get(URL + "get_avg_val", headers=client_headers)
        except requests.exceptions.ConnectionError:
            time.sleep(3)
        server_val = pickle.loads(response.content)
        
        time.sleep(sleep_delay)

    return server_val

if __name__ == '__main__':
    
    script_st = time.time()
     
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
    N_EPOCHS = 250

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
    else:
        r = requests.get(URL + "kernel_init")
        k_init = pickle.loads(r.content)
    
    #################### CLIENT SETUP ####################

    client_headers = {
        "content-type": "binary tensor", 
        "site": SITE, 
        "val_type": None, # determined later in code
        "step": None, # determined later in code
    }
    
    if MODE == "federated":
        client_headers["val_type"] = "gradients"
    elif MODE == "weightavg" or MODE == "cyclic":
        client_headers["val_type"] = "weights"

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
    
    print("Full dataset count: {}".format(len(x)))

    # if training on both sets, split data in half so the model doesn't 
    # train with "double data" compared to single site
    # also cut `BATCH_SIZE` in half
    if SITE == "BOTH":
        x = x[:len(x)//2]
        y = y[:len(y)//2]
        # do not half batch size here, because
        # BOTH has double batch size of FL
    elif MODE == "federated":
        x = x[:len(x)//2]
        y = y[:len(y)//2]
        BATCH_SIZE = BATCH_SIZE // 2
        
    split = int(np.ceil(0.8 * len(x)))
    
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    
    
    print("Truncated dataset count: {}".format(len(x_train) + len(x_val)))
    
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

    N_TRAIN_STEPS = int(np.ceil(len(x_train) / BATCH_SIZE))
    N_VAL_STEPS = int(np.ceil(len(x_val) / BATCH_SIZE))
    
    
    while(True):
        
        # update epoch
        cur_epoch += 1
        epoch_st = time.time()

        # Reset metrics every epoch
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()
        
        if MODE == "weightavg" or MODE == "cyclic":
            # Reset optimizer every epoch
            opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
            
            # keep copy of weights before local training 
            prev_weights = [layer.numpy().copy() for layer in model.trainable_variables]

        #################### TRAINING ####################

        for cur_step, i in enumerate(range(0, len(x_train), BATCH_SIZE)):

            st = time.time()

            xs = x_train[i:i+BATCH_SIZE]
            ys = y_train[i:i+BATCH_SIZE]
            
            # Logits == predictions
            client_grads, loss, preds = forward(
                inputs=(xs, ys),
                model=model,
                loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits,
                training=True,
            )
            
            '''
            If federated, then get synchronized gradient updates from server
            
            If weightavg, then apply normal gradients. Federation occurs at end of epoch.
            
            If cyclic, then apply normal gradients. Federation occurs at end of epoch.
            
            If local, then apply normal gradients and train as usual
            
            '''

            if MODE == "federated":
                grads = federate_vals(URL, client_grads, client_headers)
            else:
                grads = client_grads
            
      
            opt.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss.update_state(loss)
            train_acc.update_state(ys, tf.argmax(preds, axis=1))
            

            #################### END-OF-STEP CALCULATIONS ####################
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=global_train_step)
                tf.summary.scalar('train_acc', train_acc.result(), step=global_train_step)
            global_train_step += 1
            
            '''
            Update header.
            
            If federated, sync on every step (ie: batch).
            
            If weightavg, sync on every epoch
            
            If cyclic, sync on every epoch
            
            Note that updating this header value multiple times per epoch in the
            cases `weightavg` and `cyclic` is fine.
            
            If local, headers aren't sent anywhere.
            '''
            if mode == "federated":
                client_headers["step"] = str(global_train_step)
            else:
                client_headers["step"] = str(cur_epoch)

            en = time.time()
            elapsed = running_average(elapsed, en-st, cur_step+1)
            eta = (N_TRAIN_STEPS - cur_step) * elapsed
            print(
                TEMPLATE.format(
                        "Training",
                        cur_epoch,
                        cur_step+1,
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
                        cur_step+1,
                        N_VAL_STEPS,
                        global_val_step,
                        elapsed,
                        eta,
                ),
                end="",
            )

        #################### END-OF-EPOCH CALCULATIONS ####################
        
        '''
        Federate `weightavg` and `cyclic` modes.
        
        '''
        if MODE == "weightavg":
            # compute weight difference before/after update
            diff_weights = [a - b for (a, b) in zip(model.trainable_variables, prev_weights)]
            # send weights to server and await new weights
            server_weights = federate_vals(URL, diff_weights, client_headers)
            # update with new server weights
            for local_layer, server_layer_weights in zip(model.trainable_variables, server_weights):
                local_layer.assign(server_layer_weights)
        elif MODE == "cyclic":
            client_weights = model.trainable_variables
            # send weights to server and await new weights
            server_weights = federate_vals(URL, client_weights, client_headers)
            # update with new server weights
            for local_layer, server_layer_weights in zip(model.trainable_variables, server_weights):
                local_layer.assign(server_layer_weights)
            
        # Elapsed epoch time
        epoch_en = time.time()
        print("\n\tEpoch elapsed time: {:.2f}s".format(epoch_en-epoch_st))
        
        # Convergence criteria
        if cur_epoch >= N_EPOCHS:
            model.save_weights(str(WEIGHT_DIR / "epoch_{}_weights.h5".format(N_EPOCHS)))
            script_en = time.time()
            print("\n*****Training elapsed time: {:.2f}s*****".format(script_en-script_st))
            sys.exit()
        
