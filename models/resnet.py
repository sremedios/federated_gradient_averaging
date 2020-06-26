import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    add,
    Dropout,
    Activation,
    Dense,
    concatenate,
)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float64')

def get_kernel_initializer(s=0):
    while True:
        yield glorot_uniform(seed=s)
        s += 1


def residual_block(prev_layer, repetitions, num_filters, k_iter):

    shortcut = prev_layer
    # Need to activate bc projection is a conv
    shortcut = Activation('relu')(shortcut)
    shortcut = Conv2D(
            filters=num_filters,
            kernel_size=1, 
            strides=1, 
            padding='same',
            kernel_initializer=next(k_iter),
    )(shortcut)


    block = shortcut

    for i in range(repetitions):
        x = Conv2D(
                filters=num_filters,
                kernel_size=3, 
                strides=1, 
                padding='same',
                kernel_initializer=next(k_iter),
        )(block)
        x = Activation('relu')(x)
        x = Conv2D(
                filters=num_filters,
                kernel_size=3, 
                strides=1, 
                padding='same',
                kernel_initializer=next(k_iter),
        )(x)

        x = add([x, block])
        block = Activation('relu')(x)

    return block 

def resnet18(k_init, n_classes, n_channels, ds=1):
    k_iter = iter(k_init)

    inputs = Input(shape=(None, None, n_channels))

    x = Conv2D(
        filters=64//ds, 
        kernel_size=7, 
        kernel_initializer=next(k_iter),
        strides=2, 
        padding='same'
    )(inputs)
    block_0 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    block_1 = residual_block(
        prev_layer=block_0, 
        repetitions=2, 
        num_filters=64//ds, 
        k_iter=k_iter,
    )
    block_2 = residual_block(
        prev_layer=block_1, 
        repetitions=2, 
        num_filters=64//ds,
        k_iter=k_iter,
    )
    block_3 = residual_block(
        prev_layer=block_2, 
        repetitions=2, 
        num_filters=64//ds,
        k_iter=k_iter,
    )
    block_4 = residual_block(
        prev_layer=block_3, 
        repetitions=2, 
        num_filters=64//ds,
        k_iter=k_iter,
    )

    x = GlobalAveragePooling2D()(block_4)
    outputs = Dense(n_classes)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def resnet34(k_init, n_classes, n_channels, ds=1):
    k_iter = iter(k_init)

    inputs = Input(shape=(None, None, n_channels))
    
    x = Conv2D(
        filters=64//ds, 
        kernel_size=7, 
        kernel_initializer=next(k_iter),
        strides=2, 
        padding='same'
    )(inputs)
    block_0 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    block_1 = residual_block(
        prev_layer=block_0, 
        repetitions=3, 
        num_filters=64//ds, 
        k_iter=k_iter,
    )
    block_2 = residual_block(
        prev_layer=block_1, 
        repetitions=4, 
        num_filters=64//ds,
        k_iter=k_iter,
    )
    block_3 = residual_block(
        prev_layer=block_2, 
        repetitions=6, 
        num_filters=64//ds,
        k_iter=k_iter,
    )
    block_4 = residual_block(
        prev_layer=block_3, 
        repetitions=3, 
        num_filters=64//ds,
        k_iter=k_iter,
    )

    x = GlobalAveragePooling2D()(block_4)
    outputs = Dense(n_classes)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
