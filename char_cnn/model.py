from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing',
      Path(__file__).resolve())

import tensorflow as tf
from tcn.tcn import TemporalConvNet

def TCN(input_layer,
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

    tcn = TemporalConvNet(
        input_layer=drop,
        num_channels=num_channels,
        sequence_length=sequence_length,
        kernel_size=kernel_size,
        dropout=dropout,
        init=init)

    decoder_b = tf.get_variable('b_h', shape=[output_size], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    decoder = tf.nn.bias_add(tf.nn.convolution(tcn, tf.expand_dims(tf.transpose(embeddings),axis=0), 'SAME'), decoder_b)

    return decoder

