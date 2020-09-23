
import tensorflow as tf

### Specific for the the Goolge Cloud Platform
######################################################################
# Insert the project id and the Bucket!
Project = 'my-project-20200813'
Bucket = 'earth-engine-bucket-1'
# Specify names of output locations in Cloud Storage.
Folder = 'ai_platform_train/unet_256_l8l7_50epoch'
Job_Dir = 'gs://' + Bucket + '/' + Folder
Model_Dir = Job_Dir + '/model'
Logs_Dir = Job_Dir + '/logs'
# Put the EEified model next to the trained model directory.
EEified_Dir = Job_Dir + '/eeified'
# training data folder and name
Image_Folder_tra = 'MSMT_RF_impervious_traData'   # !can't write into the second-level directory
Image_Folder_eva = 'MSMT_RF_impervious_evaData'

#######################################################################

## TFRecord features
# output bands
Bands_l8 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
Bands_l57 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']
Targets = ['impervious']
Features_l8 = Bands_l8 + Targets
Features_l57 = Bands_l57 + Targets

# Specify the size and shape of patches expected by the model.
Kernel_shape = [256, 256]
Columns_l8 = [
  tf.io.FixedLenFeature(shape=Kernel_shape, dtype=tf.float32) for k in Features_l8
]
Features_Dict_l8 = dict(zip(Features_l8, Columns_l8))

Columns_l57 = [
  tf.io.FixedLenFeature(shape=Kernel_shape, dtype=tf.float32) for k in Features_l57
]

Features_Dict_l57 = dict(zip(Features_l57, Columns_l57))

# Specify model training parameters.
Batch_Size = 16
Epochs = 50
Buffer_Size = 2000
Optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9)
Loss = 'MeanSquaredError'
Metrics = ['RootMeanSquaredError']