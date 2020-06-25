import tensorflow as tf

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float64')

def get_kernel_initializer(s=0):
    while True:
        yield glorot_uniform(seed=s)
        s += 1
    
def cnn(k_init, n_channels, n_classes):
    
    k_iter = iter(k_init)
    # Initialize layers with weights
    weighted_layers = [
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Dense(n_classes, kernel_initializer=next(k_iter)),
    ]
    
    layer_iter = iter(weighted_layers)
    
    # Construct model
    inputs = Input((None, None, n_channels))
    a = next(layer_iter)(inputs)
    a = MaxPooling2D(2)(a)
    a = next(layer_iter)(a)
    a = MaxPooling2D(2)(a)
    a = next(layer_iter)(a)
    a = MaxPooling2D(2)(a)
    a = next(layer_iter)(a)
    a = GlobalMaxPooling2D()(a)
    outputs = next(layer_iter)(a)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    return model
