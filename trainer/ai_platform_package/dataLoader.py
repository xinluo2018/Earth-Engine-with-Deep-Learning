
from . import config
import tensorflow as tf
import random

# Dataset loading functions

tra_pattern_l57 = 'gs://' + config.Bucket + '/' + config.Image_Folder_tra + '/' + 'Train_Landsat7*.tfrecord.gz'
tra_pattern_l8 = 'gs://' + config.Bucket + '/' + config.Image_Folder_tra + '/' + 'Train_Landsat8*.tfrecord.gz'
eva_pattern_l57 = 'gs://' + config.Bucket + '/' + config.Image_Folder_eva + '/' + 'Eva_Landsat7*.tfrecord.gz'
eva_pattern_l8 = 'gs://' + config.Bucket + '/' + config.Image_Folder_eva + '/' + 'Eva_Landsat8*.tfrecord.gz'
print(tra_pattern_l57)

# Dataset loading functions
def parse_tfrecord_l57(example_proto):
	return tf.io.parse_single_example(example_proto, config.Features_Dict_l57)
 
def to_tuple_l57(inputs):
    inputsList = [inputs.get(key) for key in config.Features_l57]
    stacked = tf.stack(inputsList, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(config.Bands_l57)], stacked[:,:,len(config.Bands_l57):]

def parse_tfrecord_l8(example_proto):
	return tf.io.parse_single_example(example_proto, config.Features_Dict_l8)
 
def to_tuple_l8(inputs):
    inputsList = [inputs.get(key) for key in config.Features_l8]
    stacked = tf.stack(inputsList, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    return stacked[:,:,:len(config.Bands_l8)], stacked[:,:,len(config.Bands_l8):]

def image_aug(image, truth, flip = True, rot = True):
    if flip == True:
        if tf.random.uniform(()) > 0.5:
            if random.randint(1,2) == 1:  ## horizontal or vertical mirroring
                image = tf.image.flip_left_right(image)
                truth = tf.image.flip_left_right(truth)
            else: 
                image = tf.image.flip_up_down(image)
                truth = tf.image.flip_up_down(truth)
    if rot == True:
        if tf.random.uniform(()) > 0.5: 
            degree = random.randint(1,3)
            image = tf.image.rot90(image, k=degree)
            truth = tf.image.rot90(truth, k=degree)
    return image, truth

def get_training_dataset():
    ## for landsat 5&7
    glob_l57 = tf.io.gfile.glob(tra_pattern_l57)
    dataset_l57 = tf.data.TFRecordDataset(glob_l57, compression_type='GZIP')    
    dataset_l57 = dataset_l57.map(parse_tfrecord_l57)
    dataset_l57 = dataset_l57.map(to_tuple_l57)
    ## for landsat 8
    glob_l8 = tf.io.gfile.glob(tra_pattern_l8)
    dataset_l8 = tf.data.TFRecordDataset(glob_l8, compression_type='GZIP')
    dataset_l8 = dataset_l8.map(parse_tfrecord_l8)
    dataset_l8 = dataset_l8.map(to_tuple_l8)
    ## combination
    combined_dataset = dataset_l57.concatenate(dataset_l8)
    combined_dataset = combined_dataset.map(image_aug)
    combined_dataset = combined_dataset.shuffle(config.Buffer_Size).batch(config.Batch_Size).repeat()
    return combined_dataset

def get_eval_dataset():
    ## for landsat 5&7
    glob_l57 = tf.io.gfile.glob(eva_pattern_l57)
    dataset_l57 = tf.data.TFRecordDataset(glob_l57, compression_type='GZIP')    
    dataset_l57 = dataset_l57.map(parse_tfrecord_l57)
    dataset_l57 = dataset_l57.map(to_tuple_l57)
    ## for landsat 8
    glob_l8 = tf.io.gfile.glob(eva_pattern_l8)
    dataset_l8 = tf.data.TFRecordDataset(glob_l8, compression_type='GZIP')
    dataset_l8 = dataset_l8.map(parse_tfrecord_l8)
    dataset_l8 = dataset_l8.map(to_tuple_l8)
    ## combination
    combined_dataset = dataset_l57.concatenate(dataset_l8)
    combined_dataset = combined_dataset.shuffle(config.Buffer_Size).batch(1).repeat()
    return combined_dataset