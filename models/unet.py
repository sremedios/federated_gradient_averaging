import tensorflow as tf

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def get_kernel_initializer(s=0):
    while True:
        yield glorot_uniform(seed=s)
        s += 1
    
def unet(k_init, ds=1):
    
    k_iter = iter(k_init)
    # Initialize layers with weights
    weighted_layers = [
        Conv2D(64//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(64//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(128//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(128//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(256//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(256//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(512//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(512//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(1024//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(1024//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(512//ds, 2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(512//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(512//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(256//ds, 2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(256//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(256//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(128//ds, 2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(128//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(128//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),

        Conv2D(64//ds, 2, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(64//ds, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(64//ds, 3, activation='relu', 
                   padding='same', kernel_initializer=next(k_iter),),
        
        Conv2D(2, 3, activation='relu', 
               padding='same', kernel_initializer=next(k_iter),),
        Conv2D(1, 1, activation='sigmoid', 
                   padding='same', kernel_initializer=next(k_iter),),
            
    ]
    
    layer_iter = iter(weighted_layers)
    
    # Construct model
    inputs = Input((None, None, 1))
    # 64, 64
    conv1 = next(layer_iter)(inputs)
    conv1 = next(layer_iter)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 128, 128
    conv2 = next(layer_iter)(pool1)
    conv2 = next(layer_iter)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 256, 256
    conv3 = next(layer_iter)(pool2)
    conv3 = next(layer_iter)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 512, 512, dropout
    conv4 = next(layer_iter)(pool3)
    conv4 = next(layer_iter)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # 1024, 1024, dropout
    conv5 = next(layer_iter)(pool4)
    conv5 = next(layer_iter)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # 512, 512, 512
    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = next(layer_iter)(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = next(layer_iter)(merge6)
    conv6 = next(layer_iter)(conv6)

    # 256, 256, 256
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = next(layer_iter)(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = next(layer_iter)(merge7)
    conv7 = next(layer_iter)(conv7)

    # 128, 128, 128
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = next(layer_iter)(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = next(layer_iter)(merge8)
    conv8 = next(layer_iter)(conv8)

    # 64, 64, 64
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = next(layer_iter)(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = next(layer_iter)(merge9)
    conv9 = next(layer_iter)(conv9)

    # 2, 1
    conv9 = next(layer_iter)(conv9)
    conv10 = next(layer_iter)(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    return model
