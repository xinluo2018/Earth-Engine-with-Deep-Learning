
try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf

############## U-Net
###  Define the downsample function
##   Conv2D+BN+ReLU
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
#     result.add(tf.keras.layers.LeakyReLU())
    result.add(tf.keras.layers.ReLU())
    return result

### Define the upsample function
##  TransposeConv2D+BN+ReLU
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

## Simple U-Net
def UNet(input_shape, nclasses=2):
    ## encoder of the U-Net
    (img_height, img_width, img_channel) = input_shape
    down_stack = [
        downsample(12, 3), # outp: (bs, img_height/2, img_width/2, 32)
        downsample(24, 3), # (bs, img_height/4, img_width/4, 64)
        downsample(48, 3), # (bs, img_height/8, img_width/8, 128)
        downsample(96, 3), # (bs, img_height/16, img_width/16, 256)
        downsample(96, 3), # (bs, img_height/32, img_width/32, 512)
        # downsample(96, 3), # (bs, img_height/64, img_width/64, 512)
        # downsample(96, 3), # (bs, img_height/128, img_width/128, 512)
    ]

    ## decoder of the U-Net
    up_stack = [
        # upsample(96, 3), # outp: (bs, img_height/64, img_width/64, 1024)
        # upsample(96, 3), # (bs, img_height/32, img_width/32, 1024)
        upsample(96, 3), # (bs, img_height/16, img_width/16, 1024)
        upsample(48, 3), # (bs, img_height/8, img_width/8, 512)
        upsample(24, 3), # (bs, img_height/4, img_width/4, 256)
        upsample(12, 3), # (bs, img_height/2, img_width/2, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    
    # define the input and output tensors
    inputs = tf.keras.layers.Input(shape=[img_height, img_width, img_channel])
    if nclasses == 2:
        last = tf.keras.layers.Conv2DTranspose(1, 3,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            activation= 'sigmoid')  ## 
    else:
        last = tf.keras.layers.Conv2DTranspose(nclasses, 3,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            activation= 'softmax')  ##
    concat = tf.keras.layers.Concatenate()    
    x = inputs
    # Downsampling through the model
    skips = []   # reserve the output of medium output of the encoder network 
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])  #  
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)