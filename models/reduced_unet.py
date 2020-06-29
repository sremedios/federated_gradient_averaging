import tensorflow as tf

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Dropout, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def get_kernel_initializer(s=0):
    while True:
        yield glorot_uniform(seed=s)
        s += 1
    
def reduced_unet(k_init, ds=1):
    
    k_iter = iter(k_init)
    # Initialize layers with weights
    weighted_layers = [
        Conv2D(64//ds, 5, strides=2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(128//ds, 5, strides=2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(256//ds, 5, strides=2, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        Conv2D(512//ds, 5, strides=2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(1024//ds, 3, strides=2, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
       
        Conv2DTranspose(512//ds, 5, strides=2, activation='relu',
                   padding='same', kernel_initializer=next(k_iter),),
        Conv2DTranspose(256//ds, 5, strides=2, activation='relu',
                   padding='same', kernel_initializer=next(k_iter),),
        Conv2DTranspose(128//ds, 5, strides=2, activation='relu',
                   padding='same', kernel_initializer=next(k_iter),),
        Conv2DTranspose(64//ds, 5, strides=2, activation='relu',
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2DTranspose(32//ds, 5, strides=2, activation='relu',
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(1, 1, activation='sigmoid',
                   padding='same', kernel_initializer=next(k_iter),),
        
            
    ]
    
    layer_iter = iter(weighted_layers)
    
    # Construct model
    inputs = Input((None, None, 1))
    conv1 = next(layer_iter)(inputs)
    conv2 = next(layer_iter)(conv1)
    conv3 = next(layer_iter)(conv2)
    conv4 = next(layer_iter)(conv3)
    
    bottleneck = next(layer_iter)(conv4)
    
    up4 = next(layer_iter)(bottleneck)
    merge4 = concatenate([conv4, up4], axis=-1)
    up3 = next(layer_iter)(merge4)
    merge3 = concatenate([conv3, up3], axis=-1)
    up2 = next(layer_iter)(merge3)
    merge2 = concatenate([conv2, up2], axis=-1)
    up1 = next(layer_iter)(merge2)
    merge1 = concatenate([conv1, up1], axis=-1)
    
    final_up = next(layer_iter)(merge1)
    
    outputs = next(layer_iter)(final_up)

    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
