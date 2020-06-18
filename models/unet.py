from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet(shared_weights):

    layer_iterator = iter(shared_weights)

    # 0
    inputs = Input((None, None, 1))

    # 1 2 3
    conv1 = next(layer_iterator)(inputs)
    conv1 = next(layer_iterator)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 4 5 6
    conv2 = next(layer_iterator)(pool1)
    conv2 = next(layer_iterator)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 7 8 9
    conv3 = next(layer_iterator)(pool2)
    conv3 = next(layer_iterator)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 10 11 12 13
    conv4 = next(layer_iterator)(pool3)
    conv4 = next(layer_iterator)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # 14 15 16
    conv5 = next(layer_iterator)(pool4)
    conv5 = next(layer_iterator)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # 17 18 19 20 21
    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = next(layer_iterator)(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = next(layer_iterator)(merge6)
    conv6 = next(layer_iterator)(conv6)

    # 22 23 24 25 26
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = next(layer_iterator)(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = next(layer_iterator)(merge7)
    conv7 = next(layer_iterator)(conv7)

    # 27 28 29 30 31
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = next(layer_iterator)(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = next(layer_iterator)(merge8)
    conv8 = next(layer_iterator)(conv8)

    # 32 33 34 35 36
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = next(layer_iterator)(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = next(layer_iterator)(merge9)
    conv9 = next(layer_iterator)(conv9)

    # 37 38
    conv9 = next(layer_iterator)(conv9)
    conv10 = next(layer_iterator)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
