# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)


@add_arg_scope
def weightNormDauConvolution1d(x, num_filters, dilation_rate, filter_size=3, stride=[1],
                            pad='VALID', init_scale=1., init=False, gated=False,
                            counters={}, reuse=False, use_dau=True):
    """a dilated convolution with weight normalization (Salimans & Kingma 2016)
       Note that init part is NEVER used in our code
       It relates to the data-dependent init in original paper 
    # Arguments
        x: A tensor of shape [N, L, Cin]
        num_filters: number of convolution filters
        dilation_rate: dilation rate / holes
        filter_size: window / kernel width of each filter
        stride: stride in convolution
        gated: use gated linear units (Dauphin 2016) as activation
    # Returns
        A tensor of shape [N, L, num_filters]
    """
    name = get_name('weight_norm_conv1d', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # currently this part is never used
        if init:
            raise Exception('Init not implemented for dau')

        else:

            # size of input x is N, L, Cin

            def weight_dau_init(shape, dtype, partition_info):
                variance_scaling = tf.initializers.variance_scaling(scale=1 / 3.0, distribution='uniform')

                # reshape from [1,S,G,F] to [1,G,S,F] so that we can use tf.variance_scaling
                shape_adap = [shape[0],
                              shape[2],
                              shape[1],
                              shape[3]]

                w = variance_scaling(shape_adap, dtype, partition_info)

                # transpose back to original shape
                return tf.transpose(w, [0, 2, 1, 3])

            def weight_normalization(v):

                num_in_channels = int(v.get_shape()[-3])
                num_dau_units = int(v.get_shape()[-2])
                num_filters = int(v.get_shape()[-1])

                def g_init(shape, dtype, partition_info):
                    v_norm = tf.norm(tf.reshape(v, [-1, num_filters]), axis=0, ord=2)
                    return tf.reshape(v_norm, shape)

                g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                    initializer=g_init, trainable=True)

                # use weight normalization (Salimans & Kingma, 2016)
                W_normed = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(v, [0, 1, 2])

                return W_normed

            skip_conn = x

            BLUR_KERNEL_SIZE = 3
            pad_size = int((BLUR_KERNEL_SIZE - 1) / 2)
            # pad-input to the left with zeros

            if use_dau:
                def mu1_exponent(mu1):
                    return tf.exp(tf.abs(mu1)) - 1
                # dau_conv1d appears  to work faster when using 408 as width instead of 400 (despite 400 being a multiple of 16)
                x = tf.pad(x, tf.constant([[0, 0, ], [0, 0, ], [0, 0, ], [0, 8]]))

                from dau_conv import dau_conv1d
                y = dau_conv1d(x,
                               num_filters,
                               dau_units=(2, 1),
                               max_kernel_size=33,
                               weights_constraint=weight_normalization,
                               weights_initializer=weight_dau_init,
                               mu1_initializer=tf.random_uniform_initializer(
                                   minval=-30 / 2, maxval=0, dtype=tf.float32),
                               #mu1_initializer=tf.random_normal_initializer(stddev=2),
                               mu1_constraint=lambda mu: -tf.abs(mu),
                               #mu1_constraint=lambda mu: -(tf.exp(tf.abs(mu)) - 1),
                               sigma_initializer=tf.constant_initializer(0.3),
                               biases_initializer=tf.zeros_initializer(),
                               dau_unit_border_bound=0.0,
                               mu_learning_rate_factor=10000,
                               dau_aggregation_forbid_positive_dim1=True,
                               activation_fn=tf.nn.relu,
                               normalizer_fn=None,
                               data_format='NCHW')
                y = y[:, :, :, :-8]
            else:
                x = tf.pad(x, tf.constant([[0, 0, ], [0, 0, ], [0, 0, ], [pad_size, 0]]))

                x = tf.transpose(x, perm=[0, 2, 3, 1])
                x = tf.reshape(x, [-1, x.shape[2], x.shape[3]])

                y = tf.layers.conv1d(x, num_filters, kernel_size=3, activation=tf.nn.relu,
                                     data_format='channels_last',padding='same')

                y = tf.reshape(y, shape=[-1, 8, y.shape[1], y.shape[2]])
                y = tf.transpose(y, [0, 3, 1, 2])


                y = y[:, :, :, :-pad_size]

            if x.shape[1] != y.shape[1]:
                skip_conn = tf.zeros_like(y)

            # ignore right-most data value since it takes future values + add skip connection
            #return y
            #return tf.nn.relu(y + skip_conn)

            return tf.layers.conv2d(tf.concat((y,skip_conn),axis=1), num_filters, kernel_size=(1,1), activation=tf.nn.relu,
                                    data_format='channels_first', padding='same')




def TemporalDauBlock(input_layer, out_channels, filter_size, stride, dilation_rate, counters,
                  dropout, init=False, atten=False, use_highway=False, gated=False, use_dau=True):
    """temporal block in TCN (Bai 2018)
    # Arguments
        input_layer: A tensor of shape [N, L, Cin]
        out_channels: output dimension
        filter_size: receptive field of a conv. filter
        stride: same as what's need in conv. function
        dilation_rate: holes inbetween
        counters: to keep track of layer names
        dropout: prob. to drop weights

        atten: (not in TCN) add self attention block after Conv.
        use_highway: (not in TCN) use highway as residual connection
        gated: (not in TCN) use gated linear unit as activation

        init: (NEVER used) data-dependent initialization

    # Returns
        A tensor of shape [N, L, out_channels]
    """
    keep_prob = 1.0 - dropout

    in_channels = input_layer.get_shape()[1]
    name = get_name('temporal_block', counters)
    with tf.variable_scope(name):

        # num_filters is the hidden units in TCN
        # which is the number of out channels
        conv1 = weightNormDauConvolution1d(input_layer, out_channels, dilation_rate,
                                        filter_size, [stride], counters=counters,
                                        init=init, gated=gated, use_dau=use_dau)
        # set noise shape for spatial dropout
        # refer to https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=TcFQu3F0y-fy
        # shape should be [N, 1, C]
        out1 = tf.nn.dropout(conv1, keep_prob)

        conv2 = weightNormDauConvolution1d(out1, out_channels, dilation_rate,
                                           filter_size, [stride], counters=counters,
                                           init=init, gated=gated, use_dau=use_dau)
        out2 = tf.nn.dropout(conv2, keep_prob)

        # highway connetions or residual connection
        residual = None
        if in_channels != out_channels:
            W_h = tf.get_variable('W_h', [1,1, int(input_layer.get_shape()[1]), out_channels],
                                  tf.float32,
                                  tf.random_normal_initializer(stddev=0.01),
                                  trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME', data_format='NCHW'), b_h, data_format='NCHW')
        else:
            print("no residual convolution")

        res = input_layer if residual is None else residual

        return tf.nn.relu(out2 + res)

def TemporalDauConvNet(input_layer, num_channels, sequence_length, embedding_size, kernel_size=2,
                    dropout=tf.constant(0.0, dtype=tf.float32), init=False,
                    atten=False, use_highway=False, use_gated=False):
    """A stacked dilated CNN architecture described in Bai 2018
    # Arguments
        input_layer: Tensor of shape [N, L, Cin]
        num_channels: # of filters for each CNN layer
        kernel_size: kernel for every CNN layer
        dropout: channel dropout after CNN

        atten: (not in TCN) add self attention block after Conv.
        use_highway: (not in TCN) use highway as residual connection
        gated: (not in TCN) use gated linear unit as activation

        init: (NEVER used) data-dependent initialization

    # Returns
        A tensor of shape [N, L, num_channels[-1]]
    """

    # convert input layer into data appropriate for dau_conv1d i.e. [n x C x h x W] where N = n * h and h = 8
    input_layer = tf.reshape(input_layer, shape=[-1, 8, input_layer.shape[1], input_layer.shape[2]])
    input_layer = tf.transpose(input_layer, [0, 3, 1, 2])

    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):

        use_dau = True
        #use_dau = True if i > 0 else False
        #use_dau = False

        print(i)
        dilation_size = 1 # 2 ** i
        out_channels = num_channels[i]
        input_layer = TemporalDauBlock(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                 counters=counters, dropout=dropout, init=init, atten=atten, gated=use_gated, use_dau=use_dau)

    # convert back to original layout
    input_layer = tf.transpose(input_layer, perm=[0, 2, 3, 1])
    input_layer = tf.reshape( input_layer, [-1, input_layer.shape[2], input_layer.shape[3]])

    # from the last layer we only take the same number of features as used for embedding/encoding
    input_layer = input_layer[:,:,:embedding_size]

    return input_layer
