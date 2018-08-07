import os

import tensorflow as tf

from models import AlexNet, IMAGENET_MEAN
from activation_pruning import drop_filters


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_parser(augment):
    def parser(path, label):
        with tf.device('/cpu:0'):
            img = tf.read_file(path)
            img = tf.image.decode_jpeg(img)
            if augment:
                img = tf.image.resize_images(img, (310, 310))
                img = tf.random_crop(img, [227, 227, 3])
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=10)
                img = tf.image.random_contrast(img, lower=0.2, upper=1.0)
            else:
                img = tf.image.resize_images(img, (227, 227))
            img = tf.subtract(img, IMAGENET_MEAN)
            img = img[:, :, ::-1]
            return img, tf.string_to_number(label, out_type=tf.int32)
    return parser


def main():
    train_images = tf.data.TextLineDataset(
        '/home/caique/datasets/caltech101/caltech101_train.txt'
    )
    train_labels = tf.data.TextLineDataset(
        '/home/caique/datasets/caltech101/caltech101_train_labels.txt'
    )
    valid_images = tf.data.TextLineDataset(
        '/home/caique/datasets/caltech101/caltech101_valid.txt'
    )
    valid_labels = tf.data.TextLineDataset(
        '/home/caique/datasets/caltech101/caltech101_valid_labels.txt'
    )

    train_data = tf.data.Dataset.zip((train_images, train_labels))
    valid_data = tf.data.Dataset.zip((valid_images, valid_labels))
    train_data = train_data.map(get_parser(False)).shuffle(3030).batch(10)
    valid_data = valid_data.map(get_parser(False)).batch(120)

    session = tf.Session()
    model = AlexNet(101)
    saver = tf.train.Saver()
    saver.restore(session, './variables/alexnet-caltech101-78')

    dropped_filters = drop_filters(
        session, model, train_data, valid_data, drop_total=500
    )

    model.train(session, train_data, valid_data,
                epochs=20,
                lr=0.0001,
                dropped_filters=dropped_filters,
                # train_layers=['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                #               'fc6', 'fc7', 'fc8'],
                # weights_path='alexnet_weights.npy',
                variables_path='./variables/alexnet-caltech101-78',
                model_name='alexnet-caltech101-finetunned')
    session.run(model.iterator.make_initializer(valid_data))
    print('final eval: {}'.format(model.eval(session)))
    session.close()


if __name__ == '__main__':
    main()
