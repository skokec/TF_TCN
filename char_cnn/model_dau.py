from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing',
      Path(__file__).resolve())

import tensorflow as tf
from tcn.tdau import TemporalDauConvNet
from dau_conv import dau_conv2d, dau_conv1d
from dau_conv import DAUGridMean
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.framework.python.ops import arg_scope


def TCNv1_DAU(input_layer,
        output_size,
        num_channels,
        embedding_size,
        kernel_size,
        dropout,
        bn_switch=None,
        init=False):
    """
      shapes:
      input_layer: b_s, L contains the integer ID
      output_size should be vocab_size

    """
    initrange = 0.1
    keep_prob_emb = 1.0 - dropout
    sequence_length = input_layer.get_shape()[-1]
    embeddings = tf.get_variable(
        'embedding',
        shape=[output_size, embedding_size],
        dtype=tf.float32,
        initializer=tf.initializers.random_uniform(-initrange, initrange))
    embedded_input = tf.nn.embedding_lookup(embeddings, input_layer)
    drop = tf.nn.dropout(embedded_input, keep_prob_emb)

    # force to gave the same number of outputs at the last layer and
    # then TemporalDauConvNet will take only embedding_size features

    #num_channels[-1] = num_channels[0]

    tcn = TemporalDauConvNet(
        input_layer=drop,
        num_channels=num_channels,
        sequence_length=sequence_length,
        embedding_size=embedding_size,
        kernel_size=kernel_size,
        dropout=dropout,
        init=init)

    decoder_b = tf.get_variable('b_h', shape=[output_size], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    decoder = tf.nn.bias_add(tf.nn.convolution(tcn, tf.expand_dims(tf.transpose(embeddings),axis=0), 'SAME'), decoder_b)

    return decoder



def TCN_DAU(input_layer,
            output_size,
            num_channels,
            embedding_size,
            kernel_size,
            dropout,
            bn_switch=None,
            init=False):
    initrange = 0.1
    keep_prob_emb = 1.0 - dropout
    keep_prob = 1.0 - dropout
    sequence_length = input_layer.get_shape()[-1]
    embeddings = tf.get_variable(
        'embedding',
        shape=[output_size, embedding_size],
        dtype=tf.float32,
        initializer=tf.initializers.random_uniform(-initrange, initrange))
    embedded_input = tf.nn.embedding_lookup(embeddings, input_layer)
    drop = tf.nn.dropout(embedded_input, keep_prob_emb)

    # assign 8 samples to height
    print('Drop shape ', drop.shape)
    paddings_ly1 = tf.constant([[0, 0],  [kernel_size - 1, kernel_size - 1], [0, 0],])

    pad = tf.pad(drop, paddings_ly1, 'CONSTANT')
    print('Pad shape ', pad.shape)
    tcn = tf.layers.conv1d(
        inputs=pad,
        filters=256,
        kernel_size=3,
        strides=1,
        kernel_initializer=tf.glorot_normal_initializer(),
        #kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        #kernel_regularizer=regularizers.l2_regularizer(0.001),
        padding='VALID',
        data_format='channels_last')
    #tcn = tf.layers.batch_normalization(tcn, center=True, scale=True,
    #                                    epsilon=0.001, axis=-1, training=bn_switch)
    #tcn = tf.nn.relu(tcn)
    print('Conv1 shape ', tcn.shape)
    tcn = tcn[:, :400, :]
    tcn = tf.layers.dropout(tcn, keep_prob)

    def weight_dau_init(shape, dtype, partition_info):
        variance_scaling = tf.initializers.variance_scaling(scale=1 / 3.0, distribution='uniform')

        # reshape from [1,S,G,F] to [1,G,S,F] so that we can use tf.variance_scaling
        shape_adap = [shape[0],
                      shape[2],
                      shape[1],
                      shape[3]]

        w = variance_scaling(shape_adap, dtype, partition_info)

        # transpose back to original shape
        return tf.transpose(w, [0,2,1,3])

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

    def restrict_future_dau_conv1d(in_data, *args, **kvargs):
        skip_conn = in_data

        BLUR_KERNEL_SIZE = 5
        pad_size = int((BLUR_KERNEL_SIZE - 1) / 2)
        # pad-input to the left with zeros
        in_data = tf.pad(in_data, tf.constant([[0, 0, ], [0, 0, ], [0, 0, ], [pad_size, 0]]))

        out_data = dau_conv1d(in_data, *args, activation_fn=None, **kvargs)
        out_data = out_data[:, :, :, :-pad_size]

        if in_data.shape[1] != out_data.shape[1]:
            skip_conn = tf.zeros_like(out_data)

        # ignore right-most data value since it takes future values + add skip connection
        return tf.nn.relu(out_data + skip_conn)

    tcn = tf.reshape(tcn, shape=[-1, 8, tcn.shape[1], tcn.shape[2]])
    tcn = tf.transpose(tcn, [0, 3, 1, 2])
    if False:
        print('Conv1 shape ', tcn.shape)

        tcn = layers_lib.repeat(
            tcn,
            4,
            restrict_future_dau_conv1d,
            256,
            dau_units=(2, 1),
            max_kernel_size=33,
            #weights_initializer=tf.random_normal_initializer(stddev=0.001),
            weights_constraint=weight_normalization,
            weights_initializer=weight_dau_init,
            #weights_regularizer=regularizers.l2_regularizer(0.0001),
            mu1_initializer=tf.random_uniform_initializer(
                minval=-30 / 2, maxval=0, dtype=tf.float32),
            mu1_constraint=lambda mu: -tf.abs(mu),
            dau_unit_border_bound=0.0,
            mu_learning_rate_factor=100,
            #normalizer_fn=tf.layers.batch_normalization,
            normalizer_fn=None,
            normalizer_params=dict(center=True,
                                   scale=True,
                                   epsilon=0.001,
                                   axis=1,
                                   training=bn_switch),
            data_format='NCHW')

    def upsample_1d(x, axis=-1, name='upsample_1d'):
        with tf.name_scope(name=name):

            out_shape = map(lambda x: x if x is not None else -1, x.shape.as_list())
            out_shape[axis] *= 2

            x = tf.expand_dims(x, axis=axis)

            rep_shape = [1] * len(x.shape)
            rep_shape[axis] = 2

            return tf.reshape(tf.tile(x,rep_shape), shape=out_shape)

    def downsample_1d(x):
        return tf.layers.max_pooling2d(x, pool_size=(1, 1), strides=(1, 2), data_format='channels_first', padding='same')

    if True:
        norm_fn_params = dict(center=True,
                              scale=True,
                              epsilon=0.001,
                              axis=1,
                              training=bn_switch)

        conv1x1_upsample_combine_params = dict(kernel_size=(1,1),
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                               padding='same',
                                               data_format='channels_first',
                                               activation=tf.nn.relu)
        with arg_scope(
                [dau_conv1d, ],
                #weights_regularizer=regularizers.l2_regularizer(0.0001),
                #weights_initializer=tf.random_normal_initializer(stddev=0.01),
                weights_constraint=weight_normalization,
                weights_initializer=weight_dau_init,
                dau_units=(2, 1),
                max_kernel_size=33,
                mu1_constraint=lambda mu: -tf.abs(mu),
                dau_unit_border_bound=0.0,
                mu_learning_rate_factor=100,
                mu1_initializer=tf.random_uniform_initializer(minval=-30 / 2, maxval=0, dtype=tf.float32),
                biases_initializer=None,
                #normalizer_fn=tf.layers.batch_normalization,
                normalizer_fn=None,
                normalizer_params=norm_fn_params,
                data_format='NCHW'):

            tcn_down_module1 = layers_lib.repeat(tcn, 1, restrict_future_dau_conv1d, 256)

            tcn = downsample_1d(tcn_down_module1)
            tcn = tf.layers.dropout(tcn, keep_prob)
            tcn_down_module2 = layers_lib.repeat(tcn, 1, restrict_future_dau_conv1d, 256)

            tcn = downsample_1d(tcn_down_module2)
            tcn = tf.layers.dropout(tcn, keep_prob)
            tcn_down_module3 = layers_lib.repeat(tcn, 1, restrict_future_dau_conv1d, 256)

            tcn = downsample_1d(tcn_down_module3)
            tcn = tf.layers.dropout(tcn, keep_prob)
            tcn = layers_lib.repeat(tcn, 2, restrict_future_dau_conv1d, 256)
            tcn = tf.layers.dropout(tcn, keep_prob)

            tcn_up_module3 = tf.concat((upsample_1d(tcn), tcn_down_module3),axis=1)
            #tcn = tf.layers.batch_normalization(
            tcn = tf.layers.conv2d(tcn_up_module3, filters = 256,
                                                                 **conv1x1_upsample_combine_params)
            #                                    **norm_fn_params)
            tcn = layers_lib.repeat(tcn, 1, restrict_future_dau_conv1d, 256)
            tcn = tf.layers.dropout(tcn, keep_prob)

            tcn_up_module2 = tf.concat((upsample_1d(tcn), tcn_down_module2), axis=1)
            #tcn = tf.layers.batch_normalization(
            tcn = tf.layers.conv2d(tcn_up_module2, filters=256,
                                                                 **conv1x1_upsample_combine_params)
            #                                    **norm_fn_params)
            tcn = layers_lib.repeat(tcn, 1, restrict_future_dau_conv1d, 256)
            tcn = tf.layers.dropout(tcn, keep_prob)

            tcn_up_module1 = tf.concat((upsample_1d(tcn), tcn_down_module1), axis=1)
            #tcn = tf.layers.batch_normalization(
            tcn =  tf.layers.conv2d(tcn_up_module1, filters=256,
                                                                 **conv1x1_upsample_combine_params)
            #                                    **norm_fn_params)
            tcn = layers_lib.repeat(tcn, 1, restrict_future_dau_conv1d, 256)
            tcn = tf.layers.dropout(tcn, keep_prob)


    print('Conv1 shape ', tcn.shape)
    #tcn = tf.transpose(tcn, [0, 2, 1])
    tcn = tf.transpose(tcn, perm=[0, 2, 3, 1])

    tcn = tf.reshape( tcn, [-1, tcn.shape[2], tcn.shape[3]])

    # from the last layer we only take the same number of features as used for embedding/encoding
    tcn = tcn[:,:,:embedding_size]

    decoder_b = tf.get_variable('b_h', shape=[output_size], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    decoder = tf.nn.bias_add(tf.nn.convolution(tcn, tf.expand_dims(tf.transpose(embeddings),axis=0), 'SAME'), decoder_b)

    return decoder