import tensorflow as tf
import numpy as np

def soft_dice_loss(y_true, y_pred, epsilon=1):
    '''
    # if truth is all zero compute inverse of mask and pred
    if tf.reduce_sum(y_true) == 0:
        pred = 1 - y_pred
        gt = 1 - y_true
    else:
        pred = y_pred
        gt = y_true
    '''
    pred = y_pred
    gt = y_true

    axes = tuple(range(1, len(pred.shape)))
    intersection = tf.reduce_sum(pred * gt, axes)
    union = tf.reduce_sum(pred + gt, axes)
    return 1 - tf.reduce_mean((2.0 * intersection + epsilon) / (union + epsilon))

def dice_coef(y_true, y_pred, epsilon=1e-8):
    pred = np.round(y_pred)
    gt = y_true

    axes = tuple(range(1, len(y_pred.shape)))
    intersection = tf.reduce_sum(tf.round(pred) * gt, axes)
    union = tf.reduce_sum(tf.round(pred) + gt, axes)

    return tf.reduce_mean((2.0 * intersection + epsilon) / (union + epsilon))
