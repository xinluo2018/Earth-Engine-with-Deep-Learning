	\�	D��@\�	D��@!\�	D��@	=i�lO�O@=i�lO�O@!=i�lO�O@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-\�	D��@'K���@1p���D��@I�.R(��?Y�/�'��@*	��"�*A�rhc�A2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord@�PS	��@!�_e���R@)�PS	��@1�_e���R@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map@�:��A��@!+-�]�cV@)�^b,ӱ^@1��NMf�%@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map@�l+�@!�EX@)��C6"U@1���	@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2 �����@!�È7��X@)��#)�S@@1��c�(5@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map@��;���@!�XA�ĩS@)�u���>@1��SVZ@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle@��FX�@!���5FX@)a���)�?1[f�ټM�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap@���@!c����R@)���j�	�?1K��3��?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate@
��O��@!��TUcdV@)Ef.py��?1̓�5�>�?:Preprocessing2U
Iterator::Model::ForeverRepeat=�|볖�@!R�^��X@)�!��I�?1�����&W?:Preprocessing2F
Iterator::Model�����@!      Y@)��im{?1��[� CC?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 63.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	'K���@'K���@!'K���@      ��!       "	p���D��@p���D��@!p���D��@*      ��!       2      ��!       :	�.R(��?�.R(��?!�.R(��?B      ��!       J	�/�'��@�/�'��@!�/�'��@R      ��!       Z	�/�'��@�/�'��@!�/�'��@JGPU