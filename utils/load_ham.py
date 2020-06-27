import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
import os
import itertools
import time
import cv2

def normalize_img(img):
    # RGB has max val of 255
    return img / 255.

def load_preprocess_fname(fname, target_shape):
    x = np.array(Image.open(fname), dtype=np.float64)
    x = cv2.resize(x, target_shape, interpolation=cv2.INTER_LINEAR)
    x = normalize_img(x)
    return x


def get_iters(site, data_dir, class_names):
    '''
    Split data by class by site, so each site has the same
    distribution of data.
    
    Test set is identical across sites.
    '''
    
    # organize by class
    fnames = sorted([f for d in data_dir.iterdir() for f in d.iterdir()])
    fnames_by_class = {
        class_names[int(k)]:list(g) for k, g in itertools.groupby(
            fnames, 
            lambda f: f.parent.name
        )
    }
    
    
    # make train set have even numbers for each class
    def smart_split(fname_list, percent):
        projected_list_length = len(fname_list[:-int(np.floor(percent*len(fname_list)))])
        if projected_list_length % 2 != 0:
            split = int(np.floor(percent*len(fname_list)))+1
        else:
            split = int(np.floor(percent*len(fname_list)))
        return split
        
    # keep last 20% of each class for test
    fnames_by_class_test = {
           k:v[-smart_split(v, 0.2):] for k, v in fnames_by_class.items()
    }
    
    # remove last 20% of each class for train
    fnames_by_class_train = {
           k:v[:-smart_split(v, 0.2)] for k, v in fnames_by_class.items()
    }
    
    fnames_by_class_a = {
        k:v[:int(np.floor(0.5*len(v)))] for k, v in fnames_by_class_train.items()
    }
    fnames_by_class_b = {
        k:v[int(np.floor(0.5*len(v))):] for k, v in fnames_by_class_train.items()
    }
    
    if site == "test":
        fnames_by_class = fnames_by_class_test
        # get an iterator for each fname list
        fname_iters = {
            k:iter(v) for k, v in fnames_by_class.items()
        }
        return fname_iters
    
    elif site == "BOTH":
        # build a fname list which is identical to 
        # multi-site training.
        # As long as the training batch size, fnames_by_class_a, and
        # fnames_by_class_b are all multiples of 2, this will
        # result in equivalent datasets
        fnames_by_class = {k:[] for k in class_names}
        
        for k in class_names:
            for a, b in zip(fnames_by_class_a[k], fnames_by_class_b[k]):
                fnames_by_class[k].append(a)
                fnames_by_class[k].append(b)

    elif site == "A":
        fnames_by_class = fnames_by_class_a
    elif site == "B":
        fnames_by_class = fnames_by_class_b
        
    # train/val split
    # keep last 20% for validation
    fnames_by_class_val = {
       k:v[-int(0.2*len(v)):] for k, v in fnames_by_class.items()
    }
    
    # remove last 20% for training
    fnames_by_class_train = {
       k:v[:-int(0.2*len(v))] for k, v in fnames_by_class.items()
    }
    
    # Epoch is defined as seeing all the most representative class samples
    
    max_length_train = 0
    for k, v in fnames_by_class_train.items():
        if len(v) > max_length_train:
            max_length_train = len(v)
        print(k, len(v))
        
    max_length_val = 0
    for k, v in fnames_by_class_val.items():
        if len(v) > max_length_val:
            max_length_val = len(v)
        print(k, len(v))
    
    # get a cyclic iterator for training fname list
    fname_iters_train = {
        k:itertools.cycle(iter(v)) for k, v in fnames_by_class_train.items()
    }
    # get an iterator for training fname list
    fname_iters_val = {
        k:iter(v) for k, v in fnames_by_class_val.items()
    }

    
    return fname_iters_train, fname_iters_val, max_length_train, max_length_val

def get_batch(fname_iters, class_names, img_shape, batch_size):

    n_samples_per_class = batch_size // len(class_names)
    
    xs = np.empty((batch_size, *img_shape), dtype=np.float64)
    ys = np.empty((batch_size,), dtype=np.int64)
    i = 0

    for y, class_name in enumerate(class_names):
        for j in range(n_samples_per_class):
            # load and preprocess next item in iterator
            x = load_preprocess_fname(next(fname_iters[class_name]), img_shape[:-1])
            
            # add to batch
            xs[i] = x
            ys[i] = y
            i += 1

    return xs, ys

