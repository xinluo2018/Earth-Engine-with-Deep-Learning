	9��v~��@9��v~��@!9��v~��@	/�>��aD@/�>��aD@!/�>��aD@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-9��v~��@Y�����6@1rS�gfy@I�G�Ȱ��?YD�H���r@*	���(
A��Q���@2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord��K�;�j@!P��1�Q@)��K�;�j@1P��1�Q@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map	Q�p@!���4zU@)�����zA@1u��8e'@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map'jinŖq@!��%��W@)�����8@1��)f~ @:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV25F�j�r@!.V���X@)�j���T1@1��UB2@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map�\�E�k@!uG���R@)�ډ��@ @1�k_��@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle�O0�q@!��O�b�W@)4GV~��?1-@~�֜?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap@���<�j@!ɫK��Q@)j�!�
�?1y��Fx�?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::ConcatenatepA�lp@!\�`�zU@)gF?N��?1x�r�(x�?:Preprocessing2U
Iterator::Model::ForeverRepeat�[�t��r@!�IW��X@)��ڊ�e�?1�Y�Qo?:Preprocessing2F
Iterator::Model���Hŭr@!      Y@)'�����?1Y�t[Tm?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 40.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2A3.1 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Y�����6@Y�����6@!Y�����6@      ��!       "	rS�gfy@rS�gfy@!rS�gfy@*      ��!       2      ��!       :	�G�Ȱ��?�G�Ȱ��?!�G�Ȱ��?B      ��!       J	D�H���r@D�H���r@!D�H���r@R      ��!       Z	D�H���r@D�H���r@!D�H���r@JGPU