�	'/2�z}@'/2�z}@!'/2�z}@	����P@����P@!����P@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-'/2�z}@�eo)�@1���iϴb@I��FXT��?Y�A�&�s@*	�x�&��A��MbI^A2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord_&��vf@!!�ݱ��K@)_&��vf@1!�ݱ��K@:Demanded file read2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map���~ns@!Aσ�#X@)2����T@1~�_�k:@:Preprocessing2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map�c?�^l@!PI��Q@)�vKr�>B@1`�2l�&@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2S!�Wt@!e��Wt�X@)�7k�%@1�ק'K@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map`��"��g@!���I��M@)�(�[ZU%@1�A�㓀
@:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMapԂ}�yf@!~6G��K@)S!����?1���K�b�?:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle[�a/�os@!�Yi%X@)x�'-\V�?10t�����?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate�n��t_l@!!�7���Q@)��q5�+�?1 )����?:Preprocessing2F
Iterator::ModelҧU��t@!      Y@)�Os�"�?1���$0�s?:Preprocessing2U
Iterator::Model::ForeverRepeat�ɧ�t@!�m?��X@)K�X�U�?1dX�j��m?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 67.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�eo)�@�eo)�@!�eo)�@      ��!       "	���iϴb@���iϴb@!���iϴb@*      ��!       2      ��!       :	��FXT��?��FXT��?!��FXT��?B      ��!       J	�A�&�s@�A�&�s@!�A�&�s@R      ��!       Z	�A�&�s@�A�&�s@!�A�&�s@JGPU�"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter[��ݖ�?![��ݖ�?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput̨D~צ?!�`}N�?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterN�`�\��?!��k����?"-
IteratorGetNext/_1_Send��i��o�?!�����?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInput���G�k�?!Ð��?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D����5�?!N'���?"@
$model/sequential_11/conv2d_18/Conv2DConv2D)w�%!ߗ?!�N%�(�?"j
@gradient_tape/model/sequential_10/conv2d_16/Conv2DBackpropFilterConv2DBackpropFilterȾcŐ�?!�JVk��?"f
<gradient_tape/model/sequential/conv2d_1/Conv2DBackpropFilterConv2DBackpropFilterb�_�P�?!E�w��?"\
2gradient_tape/model/conv2d_17/Conv2DBackpropFilterConv2DBackpropFilter+���N;�?!�FeZt�?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 67.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 