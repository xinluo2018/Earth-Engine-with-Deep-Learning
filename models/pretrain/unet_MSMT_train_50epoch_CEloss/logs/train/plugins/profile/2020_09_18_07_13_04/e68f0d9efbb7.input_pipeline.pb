	��8�d��@��8�d��@!��8�d��@	Á�E@Á�E@!Á�E@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��8�d��@$�� "@1�%�<PHx@I��f���?YjO�9�%r@*	X9��V�	Ak���=j�@)      �=2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord�:�Bj@!���"R@)�:�Bj@1���"R@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map0�^|Q�o@!⿅�U@)�g�@@18F��+&@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map�f��vq@!0�H�%~W@)ݳ���3@1h \�q@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2��Itr@!G��Jy�X@)q!��F^1@1��b��@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map1\ �k@!�M�FS@)#��~j$@1�7��@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffler�	ۏq@!��o�W@)_z�sѐ�?1ދ�p�B�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap�-X�Dj@!����#R@)i���!�?1�AJlm�?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate�Qf�o@!*	��U@)�Wt�5=�?1&��?:Preprocessing2U
Iterator::Model::ForeverRepeat�N]��r@!t 1���X@)�[[%X�?1E�g�s?:Preprocessing2F
Iterator::Model�����r@!      Y@)Ku/3l�?1���o�4l?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 42.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$�� "@$�� "@!$�� "@      ��!       "	�%�<PHx@�%�<PHx@!�%�<PHx@*      ��!       2      ��!       :	��f���?��f���?!��f���?B      ��!       J	jO�9�%r@jO�9�%r@!jO�9�%r@R      ��!       Z	jO�9�%r@jO�9�%r@!jO�9�%r@JGPU