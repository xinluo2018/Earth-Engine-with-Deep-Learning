�	n�HJG�@n�HJG�@!n�HJG�@	w�8&Q@w�8&Q@!w�8&Q@"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-n�HJG�@O��'�9@1TR'���q@IE�[��"�?Yo,(j@�@*	�p=�4�AA5^���A2�
eIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap[2]::TFRecord ��4)�Bv@!�=Ӊ�L@)��4)�Bv@1�=Ӊ�L@:Demanded file read2l
5Iterator::Model::ForeverRepeat::BatchV2::Shuffle::Map �[t�t*�@!`ץ�W@)�_?��b@1�E{3�#8@:Preprocessing2�
JIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map wR~{@!��Y�Q@)����N@1����#@:Preprocessing2^
'Iterator::Model::ForeverRepeat::BatchV2�ƻCE�@!P���X@)/�$�A@1�x���@:Preprocessing2�
OIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map <�$�4w@!O��N@)��b.@1�R1�z@:Preprocessing2g
0Iterator::Model::ForeverRepeat::BatchV2::Shuffle yY�+�@!�|3C�W@)2v�Kp��?1XdJ�ՙ?:Preprocessing2�
XIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate[0]::Map::Map::FlatMap MN��Dv@! �����L@)���;޼?1 �p�]��?:Preprocessing2y
BIterator::Model::ForeverRepeat::BatchV2::Shuffle::Map::Concatenate ��{@!�(���Q@)��Tގp�?1�Nnv$&�?:Preprocessing2F
Iterator::Model;��E�@!      Y@)1��c�g�?1�Q�g~�s?:Preprocessing2U
Iterator::Model::ForeverRepeatds�<gE�@!�a��X@)�,D���?1l��<g?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 68.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	O��'�9@O��'�9@!O��'�9@      ��!       "	TR'���q@TR'���q@!TR'���q@*      ��!       2      ��!       :	E�[��"�?E�[��"�?!E�[��"�?B      ��!       J	o,(j@�@o,(j@�@!o,(j@�@R      ��!       Z	o,(j@�@o,(j@�@!o,(j@�@JGPU�"j
@gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropFilterConv2DBackpropFilterf6\�?!f6\�?"`
7model/sequential_11/conv2d_transpose_5/conv2d_transposeConv2DBackpropInputi�:&�?!J����?"-
IteratorGetNext/_1_Send�jd����?!�'�sQ��?"�
Zgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�ܢ�:��?!5�*��?"h
Lgradient_tape/model/sequential_11/conv2d_transpose_5/conv2d_transpose/Conv2DConv2D(f<�G�?!�k`����?"@
$model/sequential_11/conv2d_18/Conv2DConv2D�E�,�?!,����?"h
?gradient_tape/model/sequential_11/conv2d_18/Conv2DBackpropInputConv2DBackpropInput�kʅ��?!L?ky�0�?"j
@gradient_tape/model/sequential_10/conv2d_16/Conv2DBackpropFilterConv2DBackpropFilter�(���O�?!֡z2���?"w
Mgradient_tape/model/sequential_11/batch_normalization_22/FusedBatchNormGradV3FusedBatchNormGradV3�P�����?!�t2���?"w
Mgradient_tape/model/sequential_11/batch_normalization_23/FusedBatchNormGradV3FusedBatchNormGradV3ѻ��ؽ�?!�r��oO�?2blackQ      Y@"�
host�Your program is HIGHLY input-bound because 68.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 