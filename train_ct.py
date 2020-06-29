import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import datetime
import time
import json
import pickle
import requests
import operator
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import nibabel as nib
from sklearn.utils import shuffle

from models.reduced_unet import *
from models.losses import *
from utils.forward import *
from utils.misc import *
from utils.patch_ops import *

# Determinism
import random
#from tfdeterminism import patch   
#patch()                                                                         
SEED = 0                                                                        
os.environ['PYTHONHASHSEED'] = str(SEED)                                        
random.seed(SEED)                                                               
np.random.seed(SEED)                                                            
#tf.random.set_seed(SEED)

def normalize_img(x):
    # clipt, then 0 mean unit variance
    min_ct_intensity = 0 # water, accounts for CSF
    max_ct_intensity = 150 # blood + 50 for noise
    x[x <= min_ct_intensity] = 0
    x[x >= max_ct_intensity] = 0
    return x / max_ct_intensity
 

def federate_vals(URL, client_val, client_headers, sleep_delay=0.01):
    ########## SEND ##########
    put_successful = False
    while not put_successful:
        data = pickle.dumps(client_val)
        try:
            response = requests.put(
                URL + "put_val",
                data=data, 
                headers=client_headers,
            )
            put_successful = pickle.loads(response.content)
        except requests.exceptions.ConnectionError:
            time.sleep(0.01)
        
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
            time.sleep(0.01)
        server_val = pickle.loads(response.content)
        
        time.sleep(sleep_delay)

    return server_val


def get_training_pair(x_list, y_list, i_list, batch_size):
    # keep batches even split of positive and negative patches
    xs_pos = []
    ys_pos = []
    
    while len(xs_pos) < batch_size:

        # select a random subject from the list
        rand_idx = np.random.choice(len(x_list))
        
        # window selection
        i = next(i_list[rand_idx])
        
        hemorrhage_present = y_list[rand_idx][i].sum() > 0

        if hemorrhage_present:
            # Take copies for the batch to allow augmentation
            x = x_list[rand_idx][i].copy()
            y = y_list[rand_idx][i].copy()

            # random flip along X-axis
            if np.random.choice([True, False]):
                x = x[::-1, ...]
                y = y[::-1, ...]

            # random flip along Y-axis
            if np.random.choice([True, False]):
                x = x[:, ::-1, ...]
                y = y[:, ::-1, ...]

            xs_pos.append(x)
            ys_pos.append(y)

    xs = np.array(xs_pos)
    ys = np.array(ys_pos)

    xs, ys = shuffle(xs, ys, random_state=int(time.time()))

    return xs, ys


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
    PATCH_SIZE = (64, 64, 1)
    BATCH_SIZE = 128
    N_EPOCHS = 100
    LEARNING_RATE = 1e-4

    WEIGHT_DIR = Path("models/weights/CT")
    TB_LOG_DIR = Path("results/tb/CT")
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

    model = reduced_unet(k_init, ds=16)
    
    model.save_weights(str(WEIGHT_DIR / "init_weights.h5"))
     
    with open(str(MODEL_PATH), 'w') as f:
        json.dump(model.to_json(), f)
    
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    # set up metrics
    train_dice = tf.keras.metrics.Mean(name="train_dice")
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_dice = tf.keras.metrics.Mean(name="val_dice")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    
    #################### LOAD DATA ####################
    fpaths = sorted(DATA_DIR.iterdir())
    ct_fpaths = sorted([x for x in fpaths if "CT" in x.name and not x.is_dir()])
    mask_fpaths = sorted([x for x in fpaths if "mask" in x.name and not x.is_dir()])
    
    ct_vols = []
    mask_vols = []
    
    for ct_fpath, mask_fpath in tqdm(zip(ct_fpaths, mask_fpaths), total=len(ct_fpaths)):
        ct = nib.load(ct_fpath).get_fdata(dtype=np.float32)
        ct = normalize_img(ct)
        ct_vols.append(ct)

        mask = nib.load(mask_fpath).get_fdata(dtype=np.float32)
        mask_vols.append(mask)
        
        
    split = int(0.8 * len(ct_fpaths))
    
    ct_vols_train = ct_vols[:split]
    mask_vols_train = mask_vols[:split]
    
    ct_vols_val = ct_vols[split:]
    mask_vols_val = mask_vols[split:]
        
    # get patches
    patch_indices_train = []
    for ct in ct_vols_train:
        idx = get_random_patch_indices(ct.shape, PATCH_SIZE)
        patch_indices_train.append(idx)
        
    patch_indices_val = []
    for ct in ct_vols_val:
        idx = get_random_patch_indices(ct.shape, PATCH_SIZE)
        patch_indices_val.append(idx)

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

	# previous paper used 1000 patches x 32 samples per epoch
	# our method will be based on that
    N_TRAIN_STEPS = int(32000 // BATCH_SIZE)
    N_VAL_STEPS = int((32000 * 0.2) // BATCH_SIZE)
    
    
    while(True):
        
        epoch_st = time.time()

        # Reset metrics every epoch
        train_loss.reset_states()
        train_dice.reset_states()
        val_loss.reset_states()
        val_dice.reset_states()
        
        if MODE == "weightavg":
            # keep copy of weights before local training 
            prev_weights = [layer.numpy().copy() for layer in model.trainable_variables]
            # reset momentum
            opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

        #################### TRAINING ####################

        for cur_step in range(N_TRAIN_STEPS):
            
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

            xs, ys = get_training_pair(
                ct_vols_train,
                mask_vols_train, 
                patch_indices_train, 
                batch_size=BATCH_SIZE, 
            )

            # Logits == predictions
            client_grads, loss, preds = forward(
                inputs=(xs, ys),
                model=model,
                loss_fn=soft_dice_loss,
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
            train_dice.update_state(dice_coef(preds, ys))
            

            #################### END-OF-STEP CALCULATIONS ####################
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=global_train_step)
                tf.summary.scalar('train_dice', train_dice.result(), step=global_train_step)
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
        
        for cur_step in range(N_VAL_STEPS):

            st = time.time()

            xs, ys = get_training_pair(
                ct_vols_val,
                mask_vols_val, 
                patch_indices_val, 
                batch_size=BATCH_SIZE, 
            )

            # Logits == predictions
            loss, preds = forward(
                inputs=(xs, ys),
                model=model,
                loss_fn=soft_dice_loss,
                training=False,
            )
            
            val_loss.update_state(loss)
            val_dice.update_state(dice_coef(preds, ys))
            

            #################### END-OF-STEP CALCULATIONS ####################
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
            
        print()

               
        #################### END-OF-EPOCH CALCULATIONS ####################
		# write validation summary
        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss.result(), step=cur_epoch)
            tf.summary.scalar('val_dice', val_dice.result(), step=cur_epoch)
        global_val_step += 1
            
        
        
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
        
