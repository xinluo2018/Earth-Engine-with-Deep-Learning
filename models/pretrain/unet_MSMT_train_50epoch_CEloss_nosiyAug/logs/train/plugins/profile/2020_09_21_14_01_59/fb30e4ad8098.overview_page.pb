�	c�	�cR�@c�	�cR�@!c�	�cR�@	�$4�PP@�$4�PP@!�$4�PP@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-c�	�cR�@�k���P@1�8EG�ls@I+Kt�Y�?Ym�y��z�@*	y�&1�(A���MacA)      �=2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord 6���Īu@!�b-�)M@)6���Īu@1�b-�)M@:Demanded file read2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map H3M$�@!#���]X@)s��/Y�a@1 �� �7@:Preprocessing2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map -��|{@!�k�uR@)a�ri��R@1X����(@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map �+���v@!�S�z�N@)aobHN�2@1��M	@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2��ťʜ�@!�����X@)��9��-@1��R�@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle �qs*%�@!t�+_X@)�٭e2�?1�eV�p�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap �6砬u@!��b��M@)������?1�z�����?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate W�Sb}{@!�r�$vR@)D5%Y���?1�����?:Preprocessing2U
Iterator::Model::ForeverRepeatU�]ܜ�@!���*��X@)����k�?1�2�M�eW?:Preprocessing2F
Iterator::Modela4+�眂@!      Y@)��|��w?1�m,��O?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 65.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�k���P@�k���P@!�k���P@      ��!       "	�8EG�ls@�8EG�ls@!�8EG�ls@*      ��!       2      ��!       :	+Kt�Y�?+Kt�Y�?!+Kt�Y�?B      ��!       J	m�y��z�@m�y��z�@!m�y��z�@R      ��!       Z	m�y��z�@m�y��z�@!m�y��z�@JGPU�"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter�46ңq�?!�46ңq�?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput���Z�?!2!���?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�1��F�?!hg�J��?"-
IteratorGetNext/_1_Send����0ɡ?!#P?S�S�?"@
$model/sequential_11/conv2d_18/Conv2DConv2D�b��?!��E��?"j
@gradient_tape/model/sequential_10/conv2d_16/Conv2DBackpropFilterConv2DBackpropFiltern �Lē�?!�}T�^��?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInput�i��Z�?!�_;�?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D&�qO�!�?!�V�%��?"f
<gradient_tape/model/sequential/conv2d_1/Conv2DBackpropFilterConv2DBackpropFilter���Eɖ?!�}��I�?"\
2gradient_tape/model/conv2d_17/Conv2DBackpropFilterConv2DBackpropFilter�e�&%�?!D�}��?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 65.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 