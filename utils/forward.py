import tensorflow as tf
import contextlib

def forward(inputs, model, loss_fn, metric_fn, training=False):
    imgs, true_masks = inputs

    if training:
        context = tf.GradientTape()
    else:
        context = contextlib.nullcontext()

    with context as tape:
        pred_masks = model(imgs, training=training)
        loss = tf.reduce_mean(loss_fn(true_masks, pred_masks))
        dice = tf.reduce_mean(metric_fn(true_masks, pred_masks))

    if training:
        grad = tape.gradient(loss, model.trainable_variables)
        return grad, loss, dice, pred_masks

    return loss, dice, pred_masks