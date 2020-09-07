
from . import config
import tensorflow as tf

# Dataset loading functions
def parse_tfrecord(example_proto):
  return tf.io.parse_single_example(example_proto, config.Features_Dict)

def to_tuple(inputs):
  inputsList = [inputs.get(key) for key in config.Features]
  stacked = tf.stack(inputsList, axis=0)
  stacked = tf.transpose(stacked, [1, 2, 0])
  return stacked[:,:,:len(config.Bands)], stacked[:,:,len(config.Bands):]

def get_dataset(pattern):
	glob = tf.io.gfile.glob(pattern)
	dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
	dataset = dataset.map(parse_tfrecord)
	dataset = dataset.map(to_tuple)
	return dataset

def get_training_dataset():
	glob = 'gs://' + config.Bucket + '/' + config.Train_Data_Folder + '/' + '*'
	dataset = get_dataset(glob)
	dataset = dataset.shuffle(config.Buffer_Size).batch(config.Batch_Size).repeat()
	return dataset

# def get_eval_dataset():
# 	glob = 'gs://' + config.DATA_BUCKET + '/' + config.FOLDER + '/' + config.EVAL_BASE + '*'
# 	dataset = get_dataset(glob)
# 	dataset = dataset.batch(1).repeat()
# 	return dataset