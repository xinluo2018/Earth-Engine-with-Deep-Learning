	��ҫy@��ҫy@!��ҫy@	�!�m�O@�!�m�O@!�!�m�O@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��ҫy@V�F�_@1z�"�a@Ix��,�?Y歺��o@*	!�rh` A�E��-��@2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecorde���h�e@!0w<�P@)e���h�e@10w<�P@:Demanded file read2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map�9#J��m@!���k/NW@)��X�?@1��V��(@:Preprocessing2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map�7���i@!n�U@6T@),����<@1
D�^��&@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2qs*p@!=;v�K�X@)e��]�P1@1�iĄ��@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::MapY����Uf@!��JldQ@)��5|[@1M�9�@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::ShuffleU0*��m@!��)�OW@)4��7�°?1jC�?��?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap5a���e@!RC�;�P@)��\m���?1=�<�?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate�f��i@!�{�6T@)1a4+ۇ�?1�6f�R7�?:Preprocessing2U
Iterator::Model::ForeverRepeat"¿Zp@!��>���X@)�ZD�7�?11v�rAy?:Preprocessing2F
Iterator::Model��!��p@!      Y@)�O��0{�?1
Y�]p�s?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 63.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	V�F�_@V�F�_@!V�F�_@      ��!       "	z�"�a@z�"�a@!z�"�a@*      ��!       2      ��!       :	x��,�?x��,�?!x��,�?B      ��!       J	歺��o@歺��o@!歺��o@R      ��!       Z	歺��o@歺��o@!歺��o@JGPU