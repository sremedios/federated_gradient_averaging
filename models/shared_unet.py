import tensorflow as tf


def init_full_unet(ds):

    s = iter(range(0, 1000))
    weight_init = tf.initializers.glorot_uniform(seed=next(s))
    initial_layers = [

        tf.keras.layers.Conv2D(64//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(64//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(128//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(128//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(256//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(256//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(512//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(512//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(1024//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(1024//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(512//ds, 2, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(512//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(512//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(256//ds, 2, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(256//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(256//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(128//ds, 2, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(128//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(128//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(64//ds, 2, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(64//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(64//ds, 3, activation='relu', padding='same', kernel_initializer=weight_init,),

        tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=weight_init,),
        tf.keras.layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer=weight_init,),
    ]

    return initial_layers
