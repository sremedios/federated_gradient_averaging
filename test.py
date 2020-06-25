import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import datetime
import time
import json
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from utils.load_mnist import *

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
})

palette = sns.color_palette("Set2", n_colors=6, desat=1)

if __name__ == '__main__':
    
    #################### HYPERPARAMS / ARGS ####################

    WEIGHT_DIR = Path("models/weights")
    DATASET = sys.argv[1] # either "MNIST" or "HAM10000"
    SITE = sys.argv[2].upper()
    MODE = sys.argv[3]
    GPU_ID = sys.argv[4]
    

    WEIGHT_DIR = Path("models/weights") / DATASET
    RESULTS_DIR = Path("results") / DATASET
    MODEL_NAME = "mode_{}_site_{}".format(MODE, SITE)
    WEIGHT_DIR = WEIGHT_DIR / MODEL_NAME
    MODEL_PATH = WEIGHT_DIR / (MODEL_NAME + ".json")
    BEST_WEIGHTS = WEIGHT_DIR / ("epoch_{}_weights.h5".format(N_EPOCHS))
    

    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    #################### LOAD MODEL ####################
    
    with open(MODEL_PATH) as json_data:
        model = tf.keras.models.model_from_json(json.load(json_data))
    model.load_weights(str(BEST_WEIGHTS))
    
    #################### LOAD DATA ####################
    if DATASET == "MNIST":
        class_names = list(range(10))
        x_test, y_true = prepare_mnist("test")
    
        #################### PREDICT ####################
        logits = model(x_test, training=False)
        preds = tf.nn.softmax(logits, axis=1)
        y_pred = [tf.argmax(p).numpy() for p in preds]
    else:
        class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        img_shape = (450, 600, 3)
        fname_iters = get_iters("test", DATA_DIR, class_names)
        
        #################### PREDICT ####################
        y_pred = []
        y_true = []
        for c in class_names:
            for fname in fname_iters[c]:
                x_test = load_preprocess(fname)
                y_true.append(class_names.index(c))
                logit = model(x_test, training=False)
                pred = tf.nn.softmax(logit, axis=1)
                y_pred.append(tf.argmax(pred).numpy())

    
    bas = balanced_accuracy_score(y_true, y_pred)
    
    #################### CONFUSION MATRIX ####################
   
    cm = confusion_matrix(
        y_true, 
        y_pred,
    )
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
    )

    ax.xaxis.labelpad = 25
    ax.yaxis.labelpad = 25

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] == 0:
                txt = "0"
            else:
                txt = "{}\n{:.2f}%".format(cm[i, j], cm_norm[i, j] * 100)
            ax.text(
                j, 
                i, 
                txt,
                ha="center", 
                va="center",
                color="white" if cm_norm[i, j] > thresh else 'black',
                fontsize=14,
            )

    print("Balanced accuracy score {:.04%}".format(bas))
            
    fig.tight_layout()
    plt.title("Balanced accuracy score: {:.2%}".format(bas))
    plt.savefig(RESULTS_DIR/"cm_bas_mode_{}_site_{}.png".format(MODE, SITE))
