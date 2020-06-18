import tensorflow as tf
import numpy as np


def load_data(batch_size):
    train, test = tf.keras.datasets.mnist.load_data()
    train_images, train_labels = train
    test_images, test_labels = test

    # reshape training images
    #train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    #test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    split_idx_1 = len(train_images) // 3
    split_idx_2 = split_idx_1 * 2

    x_train_a = train_images[:split_idx_1]
    x_train_b = train_images[split_idx_1:split_idx_2]
    x_train_c = train_images[split_idx_2:]

    y_train_a = train_labels[:split_idx_1]
    y_train_b = train_labels[split_idx_1:split_idx_2]
    y_train_c = train_labels[split_idx_2:]


    split_idx_1 = len(test_images) // 3
    split_idx_2 = split_idx_1 * 2

    x_test_a = test_images[:split_idx_1]
    x_test_b = test_images[split_idx_1:split_idx_2]
    x_test_c = test_images[split_idx_2:]

    y_test_a = test_labels[:split_idx_1]
    y_test_b = test_labels[split_idx_1:split_idx_2]
    y_test_c = test_labels[split_idx_2:]

    total = len(x_test_a)

    # reshape A dataset
    y_train_a = tf.keras.utils.to_categorical(y_train_a, 10)
    y_test_a = tf.keras.utils.to_categorical(y_test_a, 10)
    #y_train_a = y_train_a.reshape(y_train_a.shape + (1,))
    #y_test_a = y_test_a.reshape(y_test_a.shape + (1,))
    
    # binarize and reshape B dataset
    y_train_b = np.array(
        [0 if y % 2 == 0 else 1 for y in y_train_b], dtype=np.float32)
    y_test_b = np.array(
        [0 if y % 2 == 0 else 1 for y in y_test_b], dtype=np.float32)
    y_train_b = y_train_b.reshape(y_train_b.shape + (1,))
    y_test_b = y_test_b.reshape(y_test_b.shape + (1,))

    # reshape C dataset
    y_train_c = np.array(y_train_c, dtype=np.float32)
    y_test_c = np.array(y_test_c, dtype=np.float32)
    y_train_c = y_train_c.reshape(y_train_c.shape + (1,))
    y_test_c = y_test_c.reshape(y_test_c.shape + (1,))

    # Create a dataset object and batch for the training data
    train_ds_a = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train_a[..., tf.newaxis]/255, tf.float32),
         tf.cast(y_train_a, tf.int64)))
    train_ds_a = train_ds_a.shuffle(1000).batch(batch_size)

    train_ds_b = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train_b[..., tf.newaxis]/255, tf.float32),
         tf.cast(y_train_b, tf.int64)))
    train_ds_b = train_ds_b.shuffle(1000).batch(batch_size)

    train_ds_c = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train_c[..., tf.newaxis]/255, tf.float32),
         tf.cast(y_train_c, tf.int64)))
    train_ds_c = train_ds_c.shuffle(1000).batch(batch_size)

    # Create a dataset object and batch for the test data
    test_ds_a = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test_a[..., tf.newaxis]/255, tf.float32),
         tf.cast(y_test_a, tf.int64)))
    test_ds_a = test_ds_a.batch(1)

    test_ds_b = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test_b[..., tf.newaxis]/255, tf.float32),
         tf.cast(y_test_b, tf.int64)))
    test_ds_b = test_ds_b.batch(1)

    test_ds_c = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test_c[..., tf.newaxis]/255, tf.float32),
         tf.cast(y_test_c, tf.int64)))
    test_ds_c = test_ds_c.batch(1)

    return train_ds_a, train_ds_b, train_ds_c, test_ds_a, test_ds_b, test_ds_c, total
