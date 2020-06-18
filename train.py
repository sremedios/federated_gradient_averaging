import pickle
import requests

import matplotlib.pyplot as plt

#from privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
#from privacy.optimizers.dp_optimizer import DPAdamOptimizer
#from privacy.optimizers.gaussian_query import GaussianAverageQuery

import json
import numpy as np
import os
from subprocess import Popen, PIPE
import sys
import time

import tensorflow as tf
import tensorflow.keras.backend as K

from utils.grad_ops import *
from utils import utils, patch_ops
from utils import preprocess

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from models.multi_gpu import ModelMGPU
from models.losses import *
from models.unet import unet

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n

if __name__ == "__main__":

    results = utils.parse_args("train")

    ########## GPU SETUP ##########

    NUM_GPUS = 1
    GPU_USAGE = 0.9

    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        # find maximum number of available GPUs
        call = "nvidia-smi --list-gpus"
        pipe = Popen(call, shell=True, stdout=PIPE).stdout
        available_gpus = pipe.read().decode().splitlines()
        NUM_GPUS = len(available_gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    #GPU_USAGE = results.GPU_USAGE

    opts = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_USAGE)
    conf = tf.ConfigProto(gpu_options=opts)
    tf.enable_eager_execution(config=conf)

    ########## SERVER SETUP ##########

    URL = "http://127.0.0.1:10203/"

    ########## HYPERPARAMETER SETUP ##########

    num_channels = results.num_channels
    plane = results.plane
    num_epochs = 1000000
    num_patches = results.num_patches
    batch_size = results.batch_size
    progbar_length = 20
    ds = 4
    model = results.model
    model_architecture = "unet"
    start_time = utils.now()
    experiment_details = start_time + "_" + model_architecture + "_" +\
        results.experiment_details
    learning_rate = 1e-4
    PATCH_SIZE = [int(x) for x in results.patch_size.split("x")]
    try:
        with open("headers.json", "r") as fp:
            headers = json.load(fp)
    except:
        print("Institution header file not present; exiting.")
        sys.exit()

    utils.save_args_to_csv(results, os.path.join(
        "results", experiment_details))

    ########## DIRECTORY SETUP ##########

    WEIGHT_DIR = os.path.join("models", "weights", experiment_details)
    TB_LOG_DIR = os.path.join("models", "tensorboard", start_time)

    MODEL_NAME = model_architecture + "_model_" + experiment_details
    MODEL_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + ".json")

    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")
    FIGURES_DIR = os.path.join("figures", MODEL_NAME)

    # files and paths
    DATA_DIR = results.SRC_DIR

    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ########## WEIGHTS AND OPTIMIZER SETUP ##########

    # shared weights from server
    r = requests.get(URL + "init_layers")
    shared_weights = pickle.loads(r.content)

    '''
    # DP optimizer
    # TODO: determine good values for DP hyperparams
    NUM_MICROBATCHES = batch_size // 4
    l2_norm_clip = 1.0
    noise_multiplier = 1.1

    dp_average_query = GaussianAverageQuery(
        l2_norm_clip=l2_norm_clip,
        sum_stddev=l2_norm_clip * noise_multiplier,
        denominator=NUM_MICROBATCHES
    )
    opt = DPAdamOptimizer(
        dp_average_query,
        NUM_MICROBATCHES,
        learning_rate=learning_rate
    )
    '''
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    ######### MODEL AND CALLBACKS #########
    model = unet(model_path=MODEL_PATH,
                 shared_weights=shared_weights,
                 num_channels=num_channels,
                 ds=ds,
                 lr=learning_rate,
                 verbose=1,)

    shared_layer_indices = []
    shared_layer_names = [l.name for l in iter(shared_weights)]
    for i, l in enumerate(model.layers):
        if l.name in shared_layer_names:
            shared_layer_indices.append(i)

    monitor = "dice_score"

    ######### PREPROCESS TRAINING DATA #########
    #DATA_DIR = os.path.join("data", "train")
    PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")

    # TODO: For now, assume preprocessing is done.
    # For some reason the files want to be preprocessed again?
    '''
    preprocess.preprocess_dir(DATA_DIR,
                              PREPROCESSED_DIR,
                              SKULLSTRIP_SCRIPT_PATH,)

    ######### DATA IMPORT #########
    ct_patches, mask_patches = patch_ops.CreatePatchesForTraining(
        atlasdir=PREPROCESSED_DIR,
        plane=plane,
        patchsize=PATCH_SIZE,
        max_patch=num_patches,
        num_channels=num_channels
    )
    '''
    ######### DATA IMPORT #########
    # TODO: this skips the preprocessed step and works directly on the
    # provided directory
    ct_patches, mask_patches = patch_ops.CreatePatchesForTraining(
        atlasdir=DATA_DIR,
        plane=plane,
        patchsize=PATCH_SIZE,
        max_patch=num_patches,
        num_channels=num_channels
    )


    # VALIDATION SPLIT
    VAL_SPLIT = int(len(ct_patches)*0.2)

    ct_patches_val = ct_patches[:VAL_SPLIT]
    mask_patches_val = mask_patches[:VAL_SPLIT]
    ct_patches = ct_patches[VAL_SPLIT:]
    mask_patches = mask_patches[VAL_SPLIT:]


    print("Individual patch dimensions:", ct_patches[0].shape)
    print("Num patches:", len(ct_patches))
    print("ct_patches shape: {}\nmask_patches shape: {}".format(
        ct_patches.shape, mask_patches.shape))

    ######### TRAINING #########
    best_val_loss = 1e6
    loss_diff = 1e6
    EARLY_STOPPING_THRESHOLD = 1e-4
    EARLY_STOPPING_EPOCHS = 50
    early_stopping_counter = 0

    dices = []
    losses = []
    val_dices = []
    val_losses = []

    def step_batch_gradient(inputs, model):
        x, y_true = inputs

        # forward pass
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(x, training=True)
            batch_loss = soft_dice_loss(y_true=y_true, y_pred=y_pred)
            batch_dice = dice_coef(y_true=y_true, y_pred=y_pred)

        batch_grads = tape.gradient(batch_loss, model.trainable_variables)

        return batch_grads, batch_loss, batch_dice

    def step_federated_grad(client_grad, num_grads):
        ########## SEND GRADIENT ##########
        r = False
        while not r:
            data = pickle.dumps((client_grad, num_grads))
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
                time.sleep(0.01)
            server_grad = pickle.loads(r.content)
            time.sleep(0.01)

        return client_grad, server_grad

    def step_batch_val(inputs, model):
        x, y_true = inputs

        y_pred = model(x, training=True)
        batch_loss = soft_dice_loss(y_true=y_true, y_pred=y_pred)
        batch_dice = dice_coef(y_true=y_true, y_pred=y_pred)

        return batch_loss, batch_dice

    ##### PROGBAR #####
    TEMPLATE = "\rEpoch {:03d}/{} [{:{}<{}}] Loss: {:>3.4f} Dice: {:>.4f}"
    sys.stdout.write(TEMPLATE.format(
        1,
        num_epochs,
        "=" * 0,
        "-",
        progbar_length,
        0.0,
        0.0,
    ))

    training_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        counter = 0
        loss = 0
        val_loss = 0
        dice_score = 0
        val_dice_score = 0

        ### TRAIN STEP ###
        num_train_steps = int(len(ct_patches))
        print("\nTraining...")
        for i in range(0, num_train_steps, batch_size):
            x = ct_patches[i: i + batch_size]
            y = mask_patches[i: i + batch_size]

            batch_grads, batch_loss, batch_dice_metric = step_batch_gradient(
                                                                 (x, y), 
                                                                 model, 
                                                             )

            # scale up grads by number of samples to allow server averaging
            scaled_batch_grads = [len(x) * g for g in batch_grads]
            # send number of samples for this batch as well
            client_grad, server_grad = step_federated_grad(scaled_batch_grads, len(x))
            
            loss += batch_loss
            dice_score += batch_dice_metric

            opt.apply_gradients(zip(server_grad, model.trainable_variables))

            sys.stdout.write(TEMPLATE.format(
                epoch + 1,
                num_epochs,
                "=" * min(
                        int(progbar_length*(i/num_train_steps)), 
                        progbar_length
                        ),
                "-",
                progbar_length,
                loss/(i+1),
                dice_score/(i+1),
            ))

        # end of epoch averages
        loss /= len(ct_patches)
        dice_score /= len(ct_patches)

        ### VAL STEP ###
        num_val_steps = int(len(ct_patches_val))
        print("\nValidating...")
        for i in range(0, num_val_steps, batch_size):
            x = ct_patches_val[i: i + batch_size]
            y = mask_patches_val[i: i + batch_size]

            batch_loss, batch_dice_metric = step_batch_val(
                                                    (x, y),
                                                    model
                                                )

            val_loss += batch_loss
            val_dice_score += batch_dice_metric

            sys.stdout.write(TEMPLATE.format(
                epoch + 1,
                num_epochs,
                "=" * min(
                        int(progbar_length*(i/num_val_steps)), 
                        progbar_length
                        ),
                "-",
                progbar_length,
                val_loss/(i+1),
                val_dice_score/(i+1),
            ))


            
        # end of epoch averages
        val_loss /= len(ct_patches_val)
        val_dice_score /= len(ct_patches_val)

        sys.stdout.write(" Val Loss: {:.4f} Val Dice: {:.4f}".format(
            val_loss,
            val_dice_score,
        ))
        print()

        ########## END OF EPOCH CALCS ##########
        epoch_end_time = time.time()
        print("Epoch time: {:.4f}s".format(epoch_end_time - epoch_start_time))

        loss_diff = np.abs(best_val_loss - val_loss)

        # early stopping and model checkpoint
        if val_loss < best_val_loss:

            # checkpoints
            checkpoint_filename = str(start_time) + "_epoch_{:04d}_".format(
                epoch) + "dice_{:.4f}_weights.hdf5".format(dice_score)

            checkpoint_filename = os.path.join(
                WEIGHT_DIR, checkpoint_filename)

            best_val_loss = val_loss
            model.save(checkpoint_filename, overwrite=True,
                       include_optimizer=False)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= EARLY_STOPPING_EPOCHS and\
                loss_diff >= EARLY_STOPPING_THRESHOLD:
            print("\nConvergence criteria reached.\nTerminating")
            break

        # metrics
        dices.append(dice_score)
        losses.append(loss)

        val_dices.append(val_dice_score)
        val_losses.append(val_loss)

    training_end_time = time.time()
    print("Training time: {:.4f}s".format(training_end_time - training_start_time))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Curves')
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].set_xlabel('Batch', fontsize=14)
    axes[0].plot(losses, label="Training Loss")
    axes[0].plot(val_losses, label="Validation Loss")

    axes[1].set_ylabel('Dice Coefficient', fontsize=14)
    axes[1].set_xlabel('Batch', fontsize=14)
    axes[1].plot(dices, label="Training Dice Coefficient")
    axes[1].plot(val_dices, label="Valdiation Dice Coefficient")

    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_curves.png'))
    K.clear_session()
