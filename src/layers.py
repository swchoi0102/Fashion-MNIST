import tensorflow as tf


def conv2d(input_, filters, training, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2d"):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(input_, filters, kernel_size=[k_h, k_w], strides=[d_h, d_w], padding='SAME',
                                kernel_initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2),
                                use_bias=False)
        bn = tf.layers.batch_normalization(conv, training=training)
        relu = tf.nn.relu(bn)

    return relu
