	�ΤM停@�ΤM停@!�ΤM停@	�SR���N@�SR���N@!�SR���N@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�ΤM停@/���	@1!yv��@I�{�i���?YL��T梑@*	�MbP��)A    ƷA)      �=2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord@nē�l$�@!Q�e�F�R@)nē�l$�@1Q�e�F�R@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map@^��\�@!^�zCV@)�'�yq`@1����Y'@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map@~�
��@!JQ��xBX@)�
cAvV@1ȗ���@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map@kׄ��?�@!g��YXS@)�����A@1�+��-�@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2���I��@!>����X@) �={�@@1���t@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle@?��%�@!q	H�ADX@)JbI���?1<r���?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap@�|�&�@!	�w�x�R@)D�+g��?1Sy�$a�?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate@����\�@!�g��,DV@)~q�J[\�?1>�ɷC�?:Preprocessing2U
Iterator::Model::ForeverRepeat��6�X��@!�&+��X@)�+�,�?1��оA9U?:Preprocessing2F
Iterator::Model�z�^��@!      Y@)eȱ��x?13��M��A?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 61.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/���	@/���	@!/���	@      ��!       "	!yv��@!yv��@!!yv��@*      ��!       2      ��!       :	�{�i���?�{�i���?!�{�i���?B      ��!       J	L��T梑@L��T梑@!L��T梑@R      ��!       Z	L��T梑@L��T梑@!L��T梑@JGPU