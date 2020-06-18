import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import datetime
import time
import json
from pathlib import Path
from tqdm import tqdm

import operator
import numpy as np
import tensorflow as tf
import nibabel as nib
from sklearn.utils import shuffle

from utils.forward import *
from utils.patch_ops import *
from utils.misc import *

from models.no_pool_unet import no_pool_unet
from models.shared_unet import *
from models.unet import *
from models.losses import soft_dice_loss, dice_coef

def get_data_pair(x_list, y_list, i_list, batch_size, thresh=10):
    # keep batches even split of positive and negative patches
    xs_pos = []
    ys_pos = []
    
    xs_neg = []
    ys_neg = []
    
    cases = [
        (xs_pos, ys_pos, operator.gt, thresh),
        #(xs_neg, ys_neg, operator.eq, 0),
    ]
    
    # For this block, `cur_xs` and `cur_ys` are pointers
    # to the `xs_pos/neg` and `ys_pos/neg` lists
    for cur_xs, cur_ys, op, val in cases:

        # trying only positive patches
        #while len(cur_xs) < batch_size//2:
        while len(cur_xs) < batch_size:

            # select a random subject from the list
            rand_idx = np.random.choice(len(x_list))

            # window selection
            i = next(i_list[rand_idx])

            # boolean conditions to use this patch
            non_empty = x_list[rand_idx][i].sum() > thresh
            # if positive case, checks for sum > thresh
            # if negative case, checks for sum == 0
            hemorrhage_cond = op(y_list[rand_idx][i].sum(), val)

            if non_empty and hemorrhage_cond:
                # Take copies for the batch to allow augmentation
                x = x_list[rand_idx][i].copy()
                y = y_list[rand_idx][i].copy()

                # random flip along X-axis
                if np.random.choice([True, False]) == True:
                    x = x[::-1, ...]
                    y = y[::-1, ...]

                # random flip along Y-axis
                if np.random.choice([True, False]) == True:
                    x = x[:, ::-1, ...]
                    y = y[:, ::-1, ...]

                cur_xs.append(x)
                cur_ys.append(y)

    xs = np.array(xs_pos + xs_neg)
    ys = np.array(ys_pos + ys_neg)

    xs, ys = shuffle(xs, ys, random_state=int(time.time()))

    return xs, ys

if __name__ == '__main__':
    
    #################### ARGUMENTS ####################
    
    DATA_DIR = Path(sys.argv[1])
    WEIGHT_DIR = Path(sys.argv[2])
    
    ds = 2
    BATCH_SIZE = 2**7
    PATCH_SIZE = (128, 128)
    learning_rate = 1e-3

    MODEL_NAME = "unet"
    experiment = "loss_{}_ds_{}_lr_{}".format(MODEL_NAME, ds, learning_rate)
    WEIGHT_DIR = WEIGHT_DIR / MODEL_NAME
    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    
    TB_LOG_DIR = Path(sys.argv[3]) / "{}_{}".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        experiment,    
    )
    
    GPU_ID = sys.argv[4]
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    
    epsilon = 1e-4
    best_val_loss = 100000 
    CONVERGENCE_EPOCH_LIMIT = 10
    convergence_epoch_counter = 0

    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not d.exists():
            d.mkdir(parents=True)
    INIT_WEIGHTS = str(WEIGHT_DIR / "init_weights.h5")

    #################### LOAD DATA ####################
    
    fpaths = sorted(DATA_DIR.iterdir())
    ct_fpaths = sorted([x for x in fpaths if "CT" in x.name and not x.is_dir()])
    mask_fpaths = sorted([x for x in fpaths if "mask" in x.name and not x.is_dir()])
    ct_fpaths, mask_fpaths = shuffle(ct_fpaths, mask_fpaths, random_state=0)
    
    # All volumes into RAM
    ct_vols = []
    mask_vols = []

    for ct_fpath, mask_fpath in tqdm(zip(ct_fpaths, mask_fpaths), total=len(ct_fpaths)):
        ct = nib.load(ct_fpath).get_fdata(dtype=np.float32)
        ct_vols.append(ct)

        mask = nib.load(mask_fpath).get_fdata(dtype=np.float32)
        mask_vols.append(mask)

    # get patch iterator
    patch_indices = []
    for ct in ct_vols:
        idx = get_random_patch_indices(ct.shape, PATCH_SIZE)
        patch_indices.append(idx)

    # 80/20 Train/Val split on subject
    split = int(.8 * len(ct_vols))

    train_ct_vols = ct_vols[:split]
    train_mask_vols = mask_vols[:split]
    train_patch_indices = patch_indices[:split]

    val_ct_vols = ct_vols[split:]
    val_mask_vols = mask_vols[split:]
    val_patch_indices = patch_indices[split:]
        
        
    #################### MODEL SETUP #################### 

    '''
    # Uncomment this to switch to getting weights from server
    # TODO: get these shared weights from a server
    '''
    shared_weights = init_full_unet(ds=ds)
    model = unet(shared_weights=shared_weights)
    
    #model = no_pool_unet(ds=ds)
    model.save_weights(INIT_WEIGHTS)
    
    print(model.summary())
        
    train_summary_writer = tf.summary.create_file_writer(str(TB_LOG_DIR / "train"))
    val_summary_writer = tf.summary.create_file_writer(str(TB_LOG_DIR / "val"))

    with open(str(MODEL_PATH), 'w') as f:
        json.dump(model.to_json(), f)
    model.load_weights(INIT_WEIGHTS)

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_dice = tf.keras.metrics.Mean(name='train_dice')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_dice = tf.keras.metrics.Mean(name='val_dice')

    #################### TRAIN #################### 
    print("\n{} TRAINING NETWORK {}\n".format(
        "#"*20,
        "#"*20,
    ))
    TEMPLATE = "\r{: >12} Epoch {: >3d} | Step {: >4d}/{: >4d} | {: >5.2f}s/step | ETA: {: >7.2f}s"
    elapsed = 0.0

    # define number of steps in an epoch
    # previous work collected 2000 patches from each image
    N_TRAIN_STEPS = 2000 * len(train_ct_vols) // BATCH_SIZE
    N_VAL_STEPS = 2000 * len(val_ct_vols) // BATCH_SIZE

    cur_epoch = 0
    global_train_step = 0
    global_val_step = 0

    while(True):
        cur_epoch += 1
        epoch_st = time.time()

        # Reset metrics every epoch
        train_loss.reset_states()
        train_dice.reset_states()
        val_loss.reset_states()
        val_dice.reset_states()

        #################### TRAINING ####################

        for cur_step in range(N_TRAIN_STEPS):

            st = time.time()

            train_batch = get_data_pair(
                train_ct_vols,
                train_mask_vols, 
                train_patch_indices, 
                batch_size=BATCH_SIZE,
            )

            # Logits are the predictions here
            grads, loss, dice, mask_preds = forward(
                inputs=train_batch,
                model=model,
                loss_fn=soft_dice_loss,
                metric_fn=dice_coef,
                training=True,
            )

            train_loss.update_state(loss)
            train_dice.update_state(dice)

            opt.apply_gradients(zip(grads, model.trainable_variables))
            
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
                        elapsed,
                        eta,
                ),
                end="",
            )

        print()

        #################### VALIDATION ####################

        for cur_step in range(N_VAL_STEPS):

            st = time.time()

            val_batch = get_data_pair(
                val_ct_vols,
                val_mask_vols, 
                val_patch_indices, 
                batch_size=BATCH_SIZE,
            )

            # Logits are the predictions here
            loss, dice, mask_preds = forward(
                inputs=val_batch,
                model=model,
                loss_fn=soft_dice_loss,
                metric_fn=dice_coef,
                training=False,
            )

            val_loss.update_state(loss)
            val_dice.update_state(dice)

            #################### END-OF-STEP CALCULATIONS ####################

            with val_summary_writer.as_default():
                tf.summary.scalar('val_loss', val_loss.result(), step=global_val_step)
                tf.summary.scalar('val_dice', val_dice.result(), step=global_val_step)

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
                        elapsed,
                        eta,
                ),
                end="",
            )

        epoch_en = time.time()
        print("\n\tEpoch elapsed time: {:.2f}s".format(epoch_en-epoch_st))

        #################### END-OF-EPOCH CALCULATIONS ####################

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
            print("\nConvergence criteria met. Terminating.\n")
            sys.exit()
