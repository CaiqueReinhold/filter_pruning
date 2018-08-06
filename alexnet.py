import tensorflow as tf

import layers


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


def alexnet_format(image, label):
    img = tf.image.resize_images(image, (227, 227))
    img = tf.cast(img, tf.float32)
    img = tf.subtract(img, IMAGENET_MEAN)
    img = img[:, :, ::-1]
    return img, label


class AlexNet(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.keep_prob = 0.5
        self.build_graph()

    def build_graph(self):
        self.iterator = tf.data.Iterator.from_structure(
            (tf.float32, tf.int32),
            (tf.TensorShape([None, 227, 227, 3]), tf.TensorShape([None]))
        )
        self.inputs, self.labels = self.iterator.get_next()

        self.conv1 = layers.conv(self.inputs, [11, 11], 96, [4, 4],
                                 padding='VALID', name='conv1')
        norm1 = layers.lrn(self.conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = layers.max_pool(norm1, [3, 3], [2, 2], padding='VALID',
                                name='pool1')

        self.conv2 = layers.conv(pool1, [5, 5], 256, [1, 1], groups=2,
                                 name='conv2')
        norm2 = layers.lrn(self.conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = layers.max_pool(norm2, [3, 3], [2, 2], padding='VALID',
                                name='pool2')

        self.conv3 = layers.conv(pool2, [3, 3], 384, [1, 1], name='conv3')

        self.conv4 = layers.conv(self.conv3, [3, 3], 384, [1, 1], groups=2,
                                 name='conv4')

        self.conv5 = layers.conv(self.conv4, [3, 3], 256, [1, 1], groups=2,
                                 name='conv5')
        pool5 = layers.max_pool(self.conv5, [3, 3], [2, 2], padding='VALID',
                                name='pool5')

        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = layers.fc(flattened, 4096, name='fc6')
        dropout6 = layers.dropout(fc6, self.keep_prob, self.training)

        fc7 = layers.fc(dropout6, 4096, name='fc7')
        dropout7 = layers.dropout(fc7, self.keep_prob, self.training)

        self.logits = layers.fc(dropout7, self.num_classes, relu=False,
                                name='fc8')
        self.probs_op = tf.nn.softmax(self.logits)
        self.pred_op = tf.argmax(input=self.logits, axis=1)
        corrects_op = tf.equal(tf.cast(self.pred_op, tf.int32),
                               self.labels)
        self.error_op = tf.reduce_mean(tf.cast(corrects_op, tf.float32))
