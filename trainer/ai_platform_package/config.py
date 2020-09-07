
import tensorflow as tf

# INSERT YOUR PROJECT HERE!
Project = 'my-project-20200813'

# INSERT YOUR BUCKET HERE!
Bucket = 'earth-engine-bucket-1'

# Specify names of output locations in Cloud Storage.
Folder = 'ai_platform_train'
Job_Dir = 'gs://' + Bucket + '/' + Folder

Model_Dir = Job_Dir + '/model'
Logs_Dir = Job_Dir + '/logs'

# Put the EEified model next to the trained model directory.
EEified_Dir = Job_Dir + '/eeified'

# Pre-computed training data.
Train_Data_Folder = 'NLCD_Impervious_Data'

# output bands
Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
Targets = ['impervious']
Features = Bands + Targets

# Specify the size and shape of patches expected by the model.
Kernel_Size = [256, 256]
Columns = [
  tf.io.FixedLenFeature(shape=Kernel_Size, dtype=tf.float32) for k in Features
]
Features_Dict = dict(zip(Features, Columns))

# Sizes of the training datasets.
Train_Size = 1000

# Specify model training parameters.
Batch_Size = 16
Epochs = 50
Buffer_Size = 2000
Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
Loss = 'MeanSquaredError'
Metrics = ['RootMeanSquaredError']