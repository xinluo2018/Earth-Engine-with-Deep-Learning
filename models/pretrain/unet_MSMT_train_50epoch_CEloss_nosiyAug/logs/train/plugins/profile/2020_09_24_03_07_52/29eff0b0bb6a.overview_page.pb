�	�'Hl��@�'Hl��@!�'Hl��@	�IlZ6S@�IlZ6S@!�IlZ6S@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�'Hl��@>\r�)@1�T��Te@IiW!�'�?Y-��Dꍁ@*	H�z��A�E��rA2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord �B����s@!l<.}L@)�B����s@1l<.}L@:Demanded file read2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map ;pΈ�d�@!��x�cW@)�O7P�0`@1��i��7@:Preprocessing2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map ���4�x@!�vn�МQ@)�*q1O@1�NEK@&@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2aE���@!�j�;�X@)UPQ�+�A@1�5�o.�@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map �Cn�[�t@!IE����M@)o���T)*@1}x���@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle \�O�e�@!�����dW@)�D2�غ?1?�I��&�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap '"��s@!�](s�~L@)����?1M�urc�?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate z8��4�x@!���VR�Q@)>�^����?1q�'�3�?:Preprocessing2U
Iterator::Model::ForeverRepeat���񵅁@!Lg����X@)ٯ;�y�?1��3�wb?:Preprocessing2F
Iterator::Model�A��Ʌ�@!      Y@)��L�σ?1F��OD\?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 76.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>\r�)@>\r�)@!>\r�)@      ��!       "	�T��Te@�T��Te@!�T��Te@*      ��!       2      ��!       :	iW!�'�?iW!�'�?!iW!�'�?B      ��!       J	-��Dꍁ@-��Dꍁ@!-��Dꍁ@R      ��!       Z	-��Dꍁ@-��Dꍁ@!-��Dꍁ@JGPU�"-
IteratorGetNext/_1_SendX� -G��?!X� -G��?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput�\��Þ�?!r��v��?"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter��OUn«?!�S���?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�哥?!UQ���?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2DDgD�A�?!�3� �?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInput���Vf��?!���X��?"@
$model/sequential_11/conv2d_18/Conv2DConv2D-�$�>��?!DUS����?"w
Mgradient_tape/model/sequential_11/batch_normalization_23/FusedBatchNormGradV3FusedBatchNormGradV3��T~�?!$�n���?"w
Mgradient_tape/model/sequential_11/batch_normalization_22/FusedBatchNormGradV3FusedBatchNormGradV3t�3jV�?!����Hq�?"`
7model/sequential_10/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput"� ��a�?!ݯi�h��?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 76.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 