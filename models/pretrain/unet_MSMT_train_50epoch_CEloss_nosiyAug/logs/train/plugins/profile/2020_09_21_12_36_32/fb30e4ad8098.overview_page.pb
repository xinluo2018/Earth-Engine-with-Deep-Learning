�	E*�-$A�@E*�-$A�@!E*�-$A�@	�\jQ�P@�\jQ�P@!�\jQ�P@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-E*�-$A�@���AB�
@1R,��<s@I6�U�š�?YU�-���@*	333��A2333��A2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord �>W[q�v@!�,8�M@)�>W[q�v@1�,8�M@:Demanded file read2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map �n�����@!ڱ�m��W@)�_ �b@1��eG�8@:Preprocessing2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map ș&l��{@!\p�ŰQ@)!Y��O@1��XH�;$@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2�S�D���@!��_���X@)`����?@1v��d@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map �uԾ�w@!T����RN@)��>r�/@1�Y'�'p@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle �4
��@!J�t�W@)����W�?1p�V��Z�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap �g$B��v@!���}�M@)��Hi6��?1H�Ū/��?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate `��5�{@!�O��V�Q@)gs�69�?1��g��?:Preprocessing2U
Iterator::Model::ForeverRepeat��͋���@!u��Z��X@)k���#G�?1�i��`?:Preprocessing2F
Iterator::Model��ɀ�@!      Y@)Y���F��?1_�Er�[?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 66.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���AB�
@���AB�
@!���AB�
@      ��!       "	R,��<s@R,��<s@!R,��<s@*      ��!       2      ��!       :	6�U�š�?6�U�š�?!6�U�š�?B      ��!       J	U�-���@U�-���@!U�-���@R      ��!       Z	U�-���@U�-���@!U�-���@JGPU�"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter���b_�?!���b_�?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput���B��?!F,-��r�?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter��gy�?!F3ډKQ�?"-
IteratorGetNext/_1_Send�>+��z�?!�����?"@
$model/sequential_11/conv2d_18/Conv2DConv2Df|:'
�?!ѹ�?�?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInput<�jH���?!B@���?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D��8XU�?!�O�gn�?"j
@gradient_tape/model/sequential_10/conv2d_16/Conv2DBackpropFilterConv2DBackpropFilter��I�|�?!)�����?"f
<gradient_tape/model/sequential/conv2d_1/Conv2DBackpropFilterConv2DBackpropFilter�n ��/�?!�!��H�?"\
2gradient_tape/model/conv2d_17/Conv2DBackpropFilterConv2DBackpropFilter*�z�Е?!���ϥ�?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 66.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 