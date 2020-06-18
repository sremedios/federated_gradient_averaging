from tensorflow.keras.layers import (
        Input,
        Conv2D, 
        Conv2DTranspose, 
        Dropout, 
        MaxPooling2D,
        concatenate,
    )
from tensorflow.keras.models import Model

def no_pool_unet(ds=2):
    inputs = Input((None, None, 1))

    conv1 = Conv2D(64//ds, 5, strides=2, activation='relu', padding='same')(inputs)
    conv2 = Conv2D(128//ds, 5, strides=2, activation='relu', padding='same')(conv1)
    conv3 = Conv2D(256//ds, 5, strides=2, activation='relu', padding='same')(conv2)
    conv4 = Conv2D(512//ds, 5, strides=2, activation='relu', padding='same')(conv3)

    bottleneck = Conv2D(1024//ds, 5, strides=2, activation='relu', padding='same')(conv4)

    up4 = Conv2DTranspose(512//ds, 5, strides=2, activation='relu', padding='same')(bottleneck)
    merge4 = concatenate([conv4, up4], axis=-1)
    up3 = Conv2DTranspose(256//ds, 5, strides=2, activation='relu', padding='same')(merge4)
    merge3 = concatenate([conv3, up3], axis=-1)
    up2 = Conv2DTranspose(128//ds, 5, strides=2, activation='relu', padding='same')(merge3)
    merge2 = concatenate([conv2, up2], axis=-1)
    up1 = Conv2DTranspose(64//ds, 5, strides=2, activation='relu', padding='same')(merge2)
    merge1 = concatenate([conv1, up1], axis=-1)

    final_up = Conv2DTranspose(32//ds, 5, strides=2, activation='relu', padding='same')(merge1)

    outputs = Conv2D(1, 1, activation='sigmoid')(final_up)

    model = Model(inputs=inputs, outputs=outputs)

    return model