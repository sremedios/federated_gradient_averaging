from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def get_kernel_initializer(s=0):
    while True:
        yield glorot_uniform(seed=s)
        s += 1
    
def cnn(k_init):
    
    k_iter = iter(k_init)
    # Initialize layers with weights
    weighted_layers = [
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=next(k_iter)),
        Dense(10, kernel_initializer=next(k_iter)),
    ]
    
    layer_iter = iter(weighted_layers)
    
    # Construct model
    inputs = Input((None, None, 1))
    a = next(layer_iter)(inputs)
    a = MaxPooling2D(2)(a)
    a = next(layer_iter)(a)
    a = MaxPooling2D(2)(a)
    a = next(layer_iter)(a)
    a = MaxPooling2D(2)(a)
    a = next(layer_iter)(a)
    a = MaxPooling2D(2)(a)
    a = GlobalAveragePooling2D()(a)
    outputs = next(layer_iter)(a)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    return model
