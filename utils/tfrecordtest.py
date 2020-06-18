import matplotlib.pyplot as plt

import numpy as np
import os

import tensorflow as tf

from utils import utils, patch_ops
from utils import preprocess

import nibabel as nib


tf.enable_eager_execution()


def image_example(X, Y):
    '''
    Creates an image example.
    X: numpy ndarray: the input image data
    Y: numpy ndarray: corresponding label information, can be an ndarray, integer, float, etc

    Returns: tf.train.Example with the following features:
        dim0, dim1, dim2, ..., dimN, X, Y, X_dtype, Y_dtype

    '''
    feature = {'dim' + str(i): tf.train.Feature(
        int64_list=tf.train.Int64List(value=[v])) for i, v in enumerate(X.shape)}
    feature['X'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[X.tobytes()]))
    feature['Y'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[Y.tobytes()]))
    feature['X_dtype'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[X.dtype.name.encode()]))
    feature['Y_dtype'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[Y.dtype.name.encode()]))

    return tf.train.Example(features=tf.train.Features(feature=feature))



##################### WRITE TF RECORD ######################


######### PREPROCESS TRAINING DATA #########
DATA_DIR = os.path.join("data")


# In[6]:


X_filenames = [os.path.join(DATA_DIR, x) for x in os.listdir(DATA_DIR)]
Y_filenames = [os.path.join(DATA_DIR, x) for x in os.listdir(DATA_DIR)]


# In[36]:


TF_RECORD_FILENAME = "mydata.tfrecords"

with tf.python_io.TFRecordWriter(TF_RECORD_FILENAME) as writer:
    # TODO: iterate over filenames here, load one at a time into RAM and then write
    for x_file, y_file in zip(X_filenames, Y_filenames):
        x = nib.load(x_file).get_fdata()
        y = nib.load(y_file).get_fdata()
        tf_example = image_example(x, y)
        writer.write(tf_example.SerializeToString())


# In[43]:





##################### READ FROM TF RECORD EXAMPLE ######################

# read from tfrecords
dataset = tf.data.TFRecordDataset(TF_RECORD_FILENAME).repeat()

# setup
dataset = dataset.map(lambda record: tf.parse_single_example(record, features={
                                                                 'dim0': tf.FixedLenFeature([], tf.int64),
    'dim1': tf.FixedLenFeature([], tf.int64),
    'dim2': tf.FixedLenFeature([], tf.int64),
    'dim3': tf.FixedLenFeature([], tf.int64),
    'X': tf.FixedLenFeature([], tf.string),
    'Y': tf.FixedLenFeature([], tf.string),
    'X_dtype': tf.FixedLenFeature([], tf.string),
    'Y_dtype': tf.FixedLenFeature([], tf.string)}
                                                            ))

data_types = {
    'uint16': tf.uint16,
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
}


# read and print
for image_features in dataset.take(1):
    ct = image_features['X']
    mask = image_features['Y']
    
    dims = [image_features[k] for k in sorted(image_features.keys()) if 'dim' in k]

    img_ct = tf.decode_raw(ct, data_types[image_features['X_dtype'].numpy().decode()])
    img_ct = np.reshape(img_ct, dims)
    img_mask = tf.decode_raw(mask, data_types[image_features['Y_dtype'].numpy().decode()])
    img_mask = np.reshape(img_mask, dims)

