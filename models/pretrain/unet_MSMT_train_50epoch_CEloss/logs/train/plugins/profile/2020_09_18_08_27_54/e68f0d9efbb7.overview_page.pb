�	�>U��{@�>U��{@!�>U��{@	���aP@���aP@!���aP@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�>U��{@��g?R@1�c�ҳ�b@I���x��?Y���~�Fr@*	�&1��	A;�O�E��@2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecordt�W�j@!Ll-{R@)t�W�j@1Ll-{R@:Demanded file read2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map����o@!�P��@�U@)���UFA@1[B�6�'@:Preprocessing2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::MapL�'�Fq@!��PǂW@)@�:s�4@14�Ī�@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2@x�^r@!b��5�X@)�º�f1@1�l&K�@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map����k@!~�`�R@)�B�i�q@1+X	�"e@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::ShuffleO���NHq@!A�����W@)0/�>:u�?1W��03�?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap�H�[�j@!��ZJ�R@)Z~�*O �?1/����?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::ConcatenateP��M�o@!fC�̹U@)�!o����?1���B�{�?:Preprocessing2F
Iterator::Model��ݒ_r@!      Y@)eo)狽�?1 j�*'p?:Preprocessing2U
Iterator::Model::ForeverRepeatc���^r@!W�Uc��X@)�릔�?1nk=zWp?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 65.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��g?R@��g?R@!��g?R@      ��!       "	�c�ҳ�b@�c�ҳ�b@!�c�ҳ�b@*      ��!       2      ��!       :	���x��?���x��?!���x��?B      ��!       J	���~�Fr@���~�Fr@!���~�Fr@R      ��!       Z	���~�Fr@���~�Fr@!���~�Fr@JGPU�"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilter�*�&�?!�*�&�?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput^�CQUѧ?!"��h��?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterK�/�?!5��u*��?"-
IteratorGetNext/_1_Send�����`�?!چZuY\�?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInputx�`R9�?!飦����?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D,���?!�b�"��?"j
@gradient_tape/model/sequential_10/conv2d_16/Conv2DBackpropFilterConv2DBackpropFilter�O��b�?!�g��R�?"f
<gradient_tape/model/sequential/conv2d_1/Conv2DBackpropFilterConv2DBackpropFiltersD*@�?!殙�T��?"@
$model/sequential_11/conv2d_18/Conv2DConv2D[�-����?!,��W��?"\
2gradient_tape/model/conv2d_17/Conv2DBackpropFilterConv2DBackpropFilter�*`����?!׌4�I�?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 65.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 