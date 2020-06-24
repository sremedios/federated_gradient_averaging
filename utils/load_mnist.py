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
    
    if site == "test":
        return (
            x_test[...,np.newaxis].astype(np.float32), 
            y_test.astype(np.int32),
        )
        
  
    # get data idx by label
    train_idx_a = np.where(y_train < 5)[0]
    train_idx_b = np.where(y_train >= 5)[0]
    
    # Balance by minimum of each
    thresh = min(len(train_idx_a), len(train_idx_b))
    train_idx_a = train_idx_a[:thresh]
    train_idx_b = train_idx_b[:thresh]
    

    if site=="BOTH":
        # build an index list which is identical to 
        # multi-site training.
        # As long as the training batch size, train_idx_a, and
        # train_idx_b are all multiples of 2, this will
        # result in equivalent datasets
        train_idx = []
        
        for a, b in zip(train_idx_a, train_idx_b):
            train_idx.append(a)
            train_idx.append(b)
        
        train_idx = np.array(train_idx)
    elif site=="A":
        train_idx = train_idx_a
    elif site=='B':
        train_idx = train_idx_b
        

    return (
        x_train[train_idx][...,np.newaxis].astype(np.float32), 
        y_train[train_idx].astype(np.int32), 
    )