import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import datetime
import time
import json
from tqdm import tqdm
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nibabel as nib

from models.losses import *
from utils.plot import *

tick_size = 20

sns.set(rc={
    'figure.figsize':(10,10), 
    'font.size': 25, 
    "axes.labelsize":25, 
    "xtick.labelsize": tick_size, 
    "ytick.labelsize": tick_size,
    'font.family':'serif',
    'grid.linestyle': '',
    'axes.facecolor': 'white',
    'axes.edgecolor': '0.2',
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'image.cmap': 'Greys_r',
    'image.interpolation': 'none',
})

palette = sns.color_palette("Set2", n_colors=6, desat=1)

def normalize_img(x):
    # clipt, then 0 mean unit variance
    min_ct_intensity = 0 # water, accounts for CSF
    max_ct_intensity = 150 # blood + 50 for noise
    x[x <= min_ct_intensity] = 0
    x[x >= max_ct_intensity] = 0
    return x / max_ct_intensity

if __name__ == '__main__':
    
    #################### HYPERPARAMS / ARGS ####################
    
    DATA_DIR = Path(sys.argv[1])
    SEG_DIR = Path(sys.argv[2])
    TEST_SITE = sys.argv[3]
    GPU_ID = sys.argv[4]

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    model_settings = [
        ("local", "A", "Local A"),
        ("local", "B", "Local B"),
        ("cyclic", "A", "Cyclic A"),
        ("cyclic", "B", "Cyclic B"),
        ("weightavg", "A", "FWA"),
        ("federated", "A", "FGA"),
    ]

    dice_df = pd.DataFrame(columns=[
        'Local A', 
        'Local B', 
        'Cyclic A', 
        'Cyclic B', 
        'FWA', 
        'FGA',
    ])
    
    #################### LOAD DATA ####################
    fpaths = sorted(DATA_DIR.iterdir())
    ct_fpaths = sorted([x for x in fpaths if "CT" in x.name and not x.is_dir()])
    mask_fpaths = sorted([x for x in fpaths if "mask" in x.name and not x.is_dir()])
    
    in_vols = []

    for ct_fpath, mask_fpath in tqdm(zip(ct_fpaths, mask_fpaths), total=len(ct_fpaths)):
        obj = nib.load(ct_fpath)
        affine = obj.affine
        header = obj.header
        ct = obj.get_fdata(dtype=np.float32)
        orig_shape = ct.shape
        ct = normalize_img(ct)
        ct = pad(ct)
        mask = nib.load(mask_fpath).get_fdata(dtype=np.float32)

        print(ct.shape)
        print(mask.shape)

        in_vols.append({
            'affine': affine,
            'header': header,
            'ct': ct,
            'mask': mask,
            'name': ct_fpath.name,
            'orig_shape': orig_shape,
        })

    sys.exit()

    #################### SEGMENT ####################
    for MODE, SITE, col_name in tqdm(model_settings):
        MODEL_NAME = "mode_{}_site_{}".format(MODE, SITE)
        WEIGHT_DIR = Path("models/weights/CT") / MODEL_NAME
        MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
        BEST_WEIGHTS = WEIGHT_DIR / "final_weights.h5"

        with open(MODEL_PATH) as json_data:
            model = tf.keras.models.model_from_json(json.load(json_data))

        model.load_weights(str(BEST_WEIGHTS))

        preds = []
        dices = []
        for in_vol in in_vols:
            # segment
            ct = in_vol['ct'].transpose(2,0,1)[..., np.newaxis]
            pred = model(ct, training=False)
            pred = pred.numpy().transpose(1,2,0,3)[:,:,:,0]
            # unpad pred to get in same space as orig
            pred = unpad(pred, in_vol['orig_shape'])
            preds.append(pred)

            # dice
            dice = dice_coef(pred, in_vol['mask']).numpy()
            dices.append(dice)

            # save nifti
            obj = nib.Nifti1Image(
                pred, 
                affine=in_vol['affine'], 
                header=in_vol['header'],
            )
            out_dir = SEG_DIR / MODEL_NAME
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            pred_fname = out_dir / in_vol['name']
            nib.save(obj, pred_fname)

        dice_df[col_name] = dices
        
    #################### PLOT ####################
    # Figure
    fig, ax = plt.subplots(1, figsize=(32,16), facecolor='white')
    # Plot
    sns.boxplot(data=dice_df)
    sns.swarmplot(data=dice_df, edgecolor="black", linewidth=1, size=15)
    # Labels
    plt.ylabel('Dice Coefficient', fontsize=20)
    plt.xlabel('')
    # Ticks
    xticks = plt.xticks()
    plt.xticks(xticks[0], fontsize=20)
    plt.yticks(np.arange(0,1.1,0.1), fontsize=20)
    # Spines
    sns.despine()
    RESULTS_DIR = Path("results")
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)
    plt.savefig(RESULTS_DIR / ("dice_coefs_testsite_{}.png".format(TEST_SITE)))
