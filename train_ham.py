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

from models.resnet import *
from utils.forward import *
from utils.misc import *
from utils.load_ham import *
from utils.opt_utils import *


'''
# Determinism
import random
from tfdeterminism import patch   
patch()                                                                         
SEED = 0                                                                        
os.environ['PYTHONHASHSEED'] = str(SEED)                                        
random.seed(SEED)                                                               
np.random.seed(SEED)                                                            
tf.random.set_seed(SEED)
'''

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
    if client_headers['mode'] == 'cyclic':
        URL_TAG = "get_cyclic_weights"
    else:
        URL_TAG = "get_avg_val"
        
    server_val = None
    while server_val is None:
        try:
            response = requests.get(URL + URL_TAG, headers=client_headers)
        except requests.exceptions.ConnectionError:
            time.sleep(3)
        server_val = pickle.loads(response.content)
        
        time.sleep(sleep_delay)

    return server_val


if __name__ == '__main__':
    
    script_st = time.time()
     
    #################### HYPERPARAMS / ARGS ####################

    SITE = sys.argv[1].upper()
    MODE = sys.argv[2]
    GPU_ID = sys.argv[3]
    DATA_DIR = Path(sys.argv[4])
    PORT = sys.argv[5]
    
    ### GPU settings ###
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    # Hyperparams 
    BATCH_SIZE = 14 # 42
    N_EPOCHS = 100   
    LEARNING_RATE = 1e-3

    RESET_MOMENTUM = False
    
    WEIGHT_DIR = Path("models/weights/HAM10000")
    TB_LOG_DIR = Path("results/tb/HAM10000")
    MODEL_NAME = "mode_{}_site_{}".format(MODE, SITE)
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

    URL = "http://0.0.0.0:{}/".format(PORT)
    if MODE == "local":
        k_init = get_kernel_initializer()
    else:
        r = requests.get(URL + "kernel_init")
        k_init = pickle.loads(r.content)
    
    #################### CLIENT SETUP ####################

    client_headers = {
        "content-type": "binary tensor", 
        "site": SITE, 
        "mode": MODE,
        "val_type": None, # determined later in code
        "step": None, # determined later in code
    }
    
    if MODE == "federated":
        client_headers["val_type"] = "gradients"
    elif MODE == "weightavg" or MODE == "cyclic":
        client_headers["val_type"] = "weights"
        

    #################### MODEL ####################

    #model = cnn(k_init, n_channels=3, n_classes=7)
    model = resnet18(k_init, n_classes=7, n_channels=3, ds=4)
    
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
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    img_shape = (450, 600, 3)
    data = get_iters(SITE, DATA_DIR, class_names)
    # unpack
    fnames_iter_train, fnames_iter_val, max_length_train, max_length_val = data

    if MODE == "federated":
        BATCH_SIZE = BATCH_SIZE // 2

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

    N_TRAIN_STEPS = int(np.ceil(max_length_train / BATCH_SIZE))
    N_VAL_STEPS = int(np.ceil(max_length_val / BATCH_SIZE))
    
    
    while(True):
        
        epoch_st = time.time()

        # Reset metrics every epoch
        train_loss.reset_states()
        train_acc.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()
        
        if RESET_MOMENTUM:
            opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        if MODE == "weightavg":
            # keep copy of weights before local training 
            prev_weights = [layer.numpy().copy() for layer in model.trainable_variables]

        #################### TRAINING ####################

        for cur_step, i in enumerate(range(0, max_length_train, BATCH_SIZE)):
            
            '''
            Update header.

            If federated, sync on every step (ie: batch).

            If weightavg, sync on every epoch

            If cyclic, sync on every epoch

            Note that updating this header value multiple times per epoch in the
            cases `weightavg` and `cyclic` is fine.

            If local, headers aren't sent anywhere.
            '''
            if MODE == "federated":
                client_headers["step"] = str(global_train_step)
                
            else:
                client_headers["step"] = str(cur_epoch)

            st = time.time()

            xs, ys = get_batch(fnames_iter_train, class_names, img_shape, BATCH_SIZE)

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
        
        # reset validation iterators
        _, fnames_iter_val, _, _ = get_iters(SITE, DATA_DIR, class_names)

        for c in class_names:
            for fname in fnames_iter_val[c]:
                st = time.time()

                x = load_preprocess_fname(fname)
                y = class_names.index(c)
                xs = x[np.newaxis,...]
                ys = np.array(y, dtype=np.int64)[np.newaxis,...]
                

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
        if MODE == "weightavg" or MODE == "cyclic":
            
            if MODE == "weightavg":
                # compute weight difference before/after update
                v = [a - b for (a, b) in zip(model.trainable_variables, prev_weights)]
                
            elif MODE == "cyclic":
                v = model.trainable_variables
                
            server_ws = federate_vals(URL, v, client_headers)
            
            # update with new server weights
            for local_w, server_w in zip(model.trainable_variables, server_ws):
                local_w.assign(server_w)

            
            
        # Elapsed epoch time
        epoch_en = time.time()
        print("\n\tEpoch elapsed time: {:.2f}s".format(epoch_en-epoch_st))
        
        # update epoch
        cur_epoch += 1
        
        # Convergence criteria
        if cur_epoch >= N_EPOCHS:
            model.save_weights(str(WEIGHT_DIR / "epoch_{}_weights.h5".format(N_EPOCHS)))
            script_en = time.time()
            print("\n*****Training elapsed time: {:.2f}s*****".format(script_en-script_st))
            sys.exit()
        
