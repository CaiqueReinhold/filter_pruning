import tensorflow as tf


def conv(x, filter_shape, num_filters, strides, name,
         padding='SAME', groups=1, mask=False):
    input_channels = int(x.shape[-1])

    def convolve(i, k):
        return tf.nn.conv2d(
            i, k,
            strides=[1] + strides + [1],
            padding=padding
        )

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable(
            'weights',
            shape=filter_shape + [input_channels // groups, num_filters],
            trainable=False
        )
        biases = tf.get_variable('biases', shape=[num_filters],
                                 trainable=False)
        tf.add_to_collection('CONV', weights)
        tf.add_to_collection('CONV', biases)

        if mask:
            mask = tf.get_variable(
                'mask',
                shape=filter_shape + [input_channels // groups, num_filters],
                trainable=False
            )
            tf.add_to_collection('MASK', mask)
            weights = weights * mask

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            output_groups = [
                convolve(i, k) for i, k in zip(input_groups, weight_groups)
            ]

            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)

        tf.add_to_collection('conv_ops', relu)

        return relu


def fc(x, units, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(
            'weights', shape=[x.shape[1], units],
            trainable=False, initializer=tf.glorot_uniform_initializer(1234)
        )
        biases = tf.get_variable('biases', shape=[units], trainable=False)
        tf.add_to_collection('FC', weights)
        tf.add_to_collection('FC', biases)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            act = tf.nn.relu(act)

        tf.add_to_collection('fc_ops', act)

        return act


def max_pool(x, filter_shape, stride_shape, name, padding='SAME'):
    return tf.nn.max_pool(
        x,
        ksize=[1] + filter_shape + [1],
        strides=[1] + stride_shape + [1],
        padding=padding,
        name=name
    )


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(
        x,
        depth_radius=radius,
        alpha=alpha, beta=beta,
        bias=bias, name=name
    )


def dropout(x, keep_prob, training=True):
    return tf.nn.dropout(x, keep_prob) if training else x
