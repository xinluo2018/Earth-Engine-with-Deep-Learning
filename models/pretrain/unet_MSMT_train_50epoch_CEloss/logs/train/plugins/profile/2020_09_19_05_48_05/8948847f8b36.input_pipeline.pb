	S����@S����@!S����@	
�˸N@
�˸N@!
�˸N@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-S����@��,�
@1iE,�х@IoF�W�@Y��7/@*	P��.�K)A&1,��A)      �=2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord@���;�@!�4k��rR@)���;�@1�4k��rR@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map@��C�@!ʨK 9DV@) ��i a@1�@	�8(@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map@-
�(J�@!0��9�AX@)��c�gSV@1�y��@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map@�]�V�@!����8=S@)�'Hlw�A@1��(��	@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2-]����@!��N��X@)�>;ຆ@@1�����@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle@S[��@!� �CX@):�l�?1>��ª�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap@�2�,�@!�<��ztR@)${��!U�?1��|౰�?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate@c��^'D�@!(V��DV@)�����?1���OA��?:Preprocessing2U
Iterator::Model::ForeverRepeat����ƌ�@!��+���X@)�i���?1KaG�Z�V?:Preprocessing2F
Iterator::Model)	��͌�@!      Y@)��ǵ�b|?1o�֨7D?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 61.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��,�
@��,�
@!��,�
@      ��!       "	iE,�х@iE,�х@!iE,�х@*      ��!       2      ��!       :	oF�W�@oF�W�@!oF�W�@B      ��!       J	��7/@��7/@!��7/@R      ��!       Z	��7/@��7/@!��7/@JGPU