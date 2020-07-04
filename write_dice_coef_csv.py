import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import datetime
import time
import json
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import numpy as np
import nibabel as nib

from models.losses import *
from utils.pad import *
from utils.plot import *

#################### HYPERPARAMS / ARGS ####################

DATA_DIR = Path(sys.argv[1])
SEG_DIR = Path(sys.argv[2])
TEST_SITE = sys.argv[3]
GPU_ID = '-1'

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
mask_fpaths = sorted([x for x in fpaths if "mask" in x.name and not x.is_dir()])

#################### CALCULATE ####################
for MODE, SITE, col_name in model_settings:
    MODEL_NAME = "mode_{}_site_{}".format(MODE, SITE)
    fpaths = sorted((SEG_DIR / MODEL_NAME).iterdir())
    pred_fpaths = sorted([x for x in fpaths if not x.is_dir()])
    
    dices = []
    for m_fpath, p_fpath in tqdm(zip(mask_fpaths, pred_fpaths), total=len(mask_fpaths)):
        mask = nib.load(m_fpath).get_fdata()
        pred = nib.load(p_fpath).get_fdata()

        mask = pad(mask)
        pred = pad(pred)

        dices.append(dice_coef(mask, pred).numpy())
    
    dice_df[col_name] = dices
    
#################### WRITE ####################
dice_df.to_csv("results/dice_coefs_test_site_{}.csv".format(TEST_SITE))
