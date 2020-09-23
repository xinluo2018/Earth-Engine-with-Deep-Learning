
import tensorflow as tf

############## U-Net
###  Define the downsample function
##   Conv2D+BN+ReLU
def downsample(filters, size, apply_dropout=True):
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                        kernel_initializer='he_normal', use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                        kernel_initializer='he_normal', use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    return result

### Define the upsample function
##  TransposeConv2D+BN+ReLU
def upsample(filters, size, apply_dropout=True):
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                        kernel_initializer='he_normal', use_bias=True))       
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                        kernel_initializer='he_normal', use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    return result

## Simple U-Net
def UNet(input_shape, nclasses=2):
    ## encoder of the U-Net
    (img_height, img_width, img_channel) = input_shape
    down_stack = [
        downsample(32, 3), # outp: (bs, img_height/2, img_width/2, 32)
        downsample(64, 3), # (bs, img_height/4, img_width/4, 64)
        downsample(64, 3), # (bs, img_height/8, img_width/8, 128)
        downsample(128, 3), # (bs, img_height/16, img_width/16, 256)
        downsample(128, 3), # (bs, img_height/32, img_width/32, 512)
        downsample(256, 3), # (bs, img_height/64, img_width/64, 512)
        # downsample(256, 3), # (bs, img_height/128, img_width/128, 512)
    ]

    ## decoder of the U-Net
    up_stack = [
        # upsample(256, 3), # output: (bs, img_height/64, img_width/64, 1024)
        upsample(256, 3), # (bs, img_height/32, img_width/32, 1024)
        upsample(128, 3), # (bs, img_height/16, img_width/16, 1024)
        upsample(64, 3), # (bs, img_height/8, img_width/8, 512)
        upsample(64, 3), # (bs, img_height/4, img_width/4, 256)
        upsample(32, 3), # (bs, img_height/2, img_width/2, 128)
    ]

    # define the input and output tensors
    inputs = tf.keras.layers.Input(shape=[img_height, img_width, img_channel])
    
    if nclasses == 2:        
        last = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same',
                    kernel_initializer='he_normal', activation= 'sigmoid')  ## (bs, 256, 256, 1)
    else:
        last = tf.keras.layers.Conv2D(nclasses, 1, strides=1, padding='same',
                    kernel_initializer='he_normal', activation= 'softmax')  ## (bs, 256, 256, 1)

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
    x = upsample(32, 3)(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)