import tensorflow as tf
import contextlib

def forward(inputs, model, loss_fn, training=False):
    xs, ys = inputs

    if training:
        context = tf.GradientTape()
    else:
        context = contextlib.nullcontext()

    with context as tape:
        preds = model(xs, training=training)
        loss = tf.reduce_mean(loss_fn(ys, preds))

    if training:
        grad = tape.gradient(loss, model.trainable_variables)
        return grad, loss, preds

    return loss, preds