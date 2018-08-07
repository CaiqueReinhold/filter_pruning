from datetime import datetime

import numpy as np
import tensorflow as tf

import layers


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class BaseModel(object):

    def __init__(self, num_classes, keep_prob=0.6):
        self.num_classes = num_classes
        self._keep_prob = keep_prob
        self.build_graph()

    def load_weights(self, session, weights_file):
        weights_dict = np.load(weights_file, encoding='bytes').item()

        op_names = [op_name for op_name in weights_dict]

        for op_name in op_names:
            with tf.variable_scope(op_name, reuse=True):
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))

    def assign_drop_masks(self, session, dropped_filters):
        masks = tf.get_collection('MASK')
        for var, dropped in zip(masks, dropped_filters):
            new_mask = np.ones(var.shape)
            for f in dropped:
                new_mask[:, :, :, f] = 0
            session.run(var.assign(new_mask))

    def train(self, session, train, valid,
              lr=0.001,
              momentum=0.7,
              epochs=30,
              train_layers=None,
              stop_with_n_steps=10,
              weights_path=None,
              variables_path=None,
              dropped_filters=None,
              model_name='model'):
        assert bool(variables_path) != bool(weights_path)
        saver = tf.train.Saver()

        one_hot = tf.stop_gradient(tf.one_hot(self.labels, self.num_classes))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot, logits=self.logits
        ))

        if train_layers is not None:
            trainable = []
            for layer in train_layers:
                trainable += tf.get_collection(layer)
        else:
            trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        opt = tf.train.MomentumOptimizer(lr, momentum)
        train_op = opt.minimize(loss, var_list=trainable)
        mean_loss = tf.reduce_mean(loss)

        train_data_init = self.iterator.make_initializer(train)
        valid_data_init = self.iterator.make_initializer(valid)

        session.run(tf.global_variables_initializer())
        session.run(self.keep_prob.assign(self._keep_prob))

        if weights_path is not None:
            self.load_weights(session, weights_path)

        if variables_path is not None:
            saver.restore(session, variables_path)

        if dropped_filters is not None:
            self.assign_drop_masks(session, dropped_filters)

        session.run(valid_data_init)
        best_accuracy = self.eval(session)
        steps_no_improve = 0
        print('train started')
        for epoch in range(epochs):
            if steps_no_improve > stop_with_n_steps:
                break

            session.run(train_data_init)
            losses = []
            try:
                while True:
                    _, loss_val = session.run([train_op, mean_loss])
                    losses.append(loss_val)
            except tf.errors.OutOfRangeError:
                pass

            session.run(valid_data_init)
            epoch_acc = self.eval(session)

            if best_accuracy < epoch_acc:
                saver.save(session, './variables/{}'.format(model_name),
                           global_step=epoch)
                best_accuracy = epoch_acc
                steps_no_improve = 0
            else:
                steps_no_improve += 1

            print(
                'Epoch {}, cost: {}, accuracy: {:3f}% - time: {}'.format(
                    epoch, np.mean(losses), epoch_acc * 100, datetime.now()
                )
            )

        print('Training complete, ran for {} epochs'.format(epoch))

    def eval(self, session):
        acc = []
        session.run(self.keep_prob.assign(1))
        try:
            while True:
                acc.append(session.run(self.acc_op))
        except tf.errors.OutOfRangeError:
            pass
        session.run(self.keep_prob.assign(self._keep_prob))
        return np.mean(acc)


class AlexNet(BaseModel):

    def build_graph(self):
        self.iterator = tf.data.Iterator.from_structure(
            (tf.float32, tf.int32),
            (tf.TensorShape([None, 227, 227, 3]), tf.TensorShape([None]))
        )
        self.inputs, self.labels = self.iterator.get_next()

        self.conv1 = layers.conv(self.inputs, [11, 11], 96, [4, 4],
                                 padding='VALID', name='conv1', mask=True)
        norm1 = layers.lrn(self.conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = layers.max_pool(norm1, [3, 3], [2, 2], padding='VALID',
                                name='pool1')

        self.conv2 = layers.conv(pool1, [5, 5], 256, [1, 1], groups=2,
                                 name='conv2', mask=True)
        norm2 = layers.lrn(self.conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = layers.max_pool(norm2, [3, 3], [2, 2], padding='VALID',
                                name='pool2')

        self.conv3 = layers.conv(pool2, [3, 3], 384, [1, 1], name='conv3',
                                 mask=True)

        self.conv4 = layers.conv(self.conv3, [3, 3], 384, [1, 1], groups=2,
                                 name='conv4', mask=True)

        self.conv5 = layers.conv(self.conv4, [3, 3], 256, [1, 1], groups=2,
                                 name='conv5', mask=True)
        pool5 = layers.max_pool(self.conv5, [3, 3], [2, 2], padding='VALID',
                                name='pool5')

        self.keep_prob = tf.get_variable('keep_prob', shape=(),
                                         trainable=False)

        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = layers.fc(flattened, 4096, name='fc6')

        fc7 = layers.fc(fc6, 4096, name='fc7')
        dropout7 = layers.dropout(fc7, self.keep_prob)

        self.logits = layers.fc(dropout7, self.num_classes, relu=False,
                                name='fc8')
        self.probs_op = tf.nn.softmax(self.logits)
        self.pred_op = tf.argmax(input=self.logits, axis=1)
        corrects_op = tf.equal(tf.cast(self.pred_op, tf.int32),
                               self.labels)
        self.acc_op = tf.reduce_mean(tf.cast(corrects_op, tf.float32))
