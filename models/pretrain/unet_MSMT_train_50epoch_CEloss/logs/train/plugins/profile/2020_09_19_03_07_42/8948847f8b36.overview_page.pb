�	emS<.R�@emS<.R�@!emS<.R�@	z��牭O@z��牭O@!z��牭O@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-emS<.R�@��+f�@1{�\�f҃@I������?Y�I��7O�@*	��|?x&'A�A`e�NA2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord@�[z���@!d%cb�Q@)�[z���@1d%cb�Q@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map@{���@!�^"QU@)��_��b@1����B+@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map@��?� �@!�x�7�KW@)w�Nyt�U@1�^���@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2��Y��N�@!�D��X@)~�D��R@1�@2�&@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map@s* *̈@!�!T���Q@)�qm�]A@1�J�n�	@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle@��%s�!�@!��#,jMW@)���7��?1�b�fFϝ?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap@TS�uX��@!�G޳ Q@)�26t�?�?1#���?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate@fM,���@!�r�QU@)��j�=�?1��KAu�?:Preprocessing2U
Iterator::Model::ForeverRepeat���O�N�@!ӏ`��X@)��ӹ���?1����W?:Preprocessing2F
Iterator::Model�!o��N�@!      Y@)�.��҈?1<-p��Q?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 63.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��+f�@��+f�@!��+f�@      ��!       "	{�\�f҃@{�\�f҃@!{�\�f҃@*      ��!       2      ��!       :	������?������?!������?B      ��!       J	�I��7O�@�I��7O�@!�I��7O�@R      ��!       Z	�I��7O�@�I��7O�@!�I��7O�@JGPU�"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter) U���?!) U���?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput�?��fĨ?!�' J�?"-
IteratorGetNext/_1_Sendbb��ڡ?!�hE��l�?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�\��e�?!�?�	$��?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInput����o�?!
��r��?"@
$model/sequential_11/conv2d_18/Conv2DConv2D/"��ܔ�?!P4����?"j
@gradient_tape/model/sequential_10/conv2d_16/Conv2DBackpropFilterConv2DBackpropFilterWv�qs=�?!�a[޵�?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D��[��?!�r>�?"w
Mgradient_tape/model/sequential_11/batch_normalization_23/FusedBatchNormGradV3FusedBatchNormGradV3���	�?!d��_U�?"w
Mgradient_tape/model/sequential_11/batch_normalization_22/FusedBatchNormGradV3FusedBatchNormGradV3�U�x��?!��n|J��?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 63.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 