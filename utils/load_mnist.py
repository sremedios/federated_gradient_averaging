import tensorflow as tf
import numpy as np

def prepare_mnist(site):
    '''
    Return all data if site=='both'
    Otherwise 'a' is [0,4]
    and 'b' is [5,9].
    
    x_test and y_test are the same regardless of site
    and are in [0,9]
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    fst_half_idx = np.where(y_train < 5)[0]
    snd_half_idx = np.where(y_train >= 5)[0]
    thresh = min(len(fst_half_idx), len(snd_half_idx))

    if site=="BOTH":
        train_idx = np.where(y_train <10)[0]
    elif site=="A":
        train_idx = np.where(y_train < 5)[0]
    elif site=='B':
        train_idx = np.where(y_train >= 5)[0]

    # This line serves two purposes:
    # 1) for sites A,B: even number of samples >=5 and < 5
    # 2) for both sites, cut samples in half (ie: same datacount as single site)
    train_idx = train_idx[:thresh]

    return (
        x_train[train_idx][...,np.newaxis].astype(np.float32), 
        y_train[train_idx].astype(np.int32), 
        x_test[...,np.newaxis].astype(np.float32), 
        y_test.astype(np.int32),
    )