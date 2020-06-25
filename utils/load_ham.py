import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
import os
import itertools
import time


def normalize_img(img):
    # RGB has max val of 255
    return img / 255.

def load_preprocess_fname(fname):
    x = np.array(Image.open(fname), dtype=np.float64)
    x = normalize_img(x)
    return x


# TODO: site differences, train/test split
def get_iters(data_dir, class_names):
    
    fnames = sorted([f for d in data_dir.iterdir() for f in d.iterdir()])
    fnames_by_class = {
        class_names[int(k)]:list(g) for k, g in itertools.groupby(
            fnames, 
            lambda f: f.parent.name
        )
    }
    
    for k, v in fnames_by_class.items():
        print(k, len(v))
    
    # get a cyclic iterator for each fname list
    fname_iters = {
        k:itertools.cycle(iter(v)) for k, v in fnames_by_class.items()
    }
    
    return fname_iters

def get_batch(fname_iters, class_names, img_shape, batch_size):

    n_samples_per_class = batch_size // len(class_names)
    
    xs = np.empty((batch_size, *img_shape), dtype=np.float64)
    ys = np.empty((batch_size,), dtype=np.int64)
    i = 0

    for y, class_name in enumerate(class_names):
        for j in range(n_samples_per_class):
            xs[i] = load_preprocess_fname(next(fname_iters[class_name]))
            ys[i] = y
            i += 1

    return xs, ys

