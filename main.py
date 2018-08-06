import os

import tensorflow as tf

from models import AlexNet, IMAGENET_MEAN


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parser(path, label):
    with tf.device('/cpu:0'):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize_images(img, (227, 227))
        img = tf.subtract(img, IMAGENET_MEAN)
        img = img[:, :, ::-1]
        return img, tf.cast(label, tf.int32)


def main():
    with open('../tf/caltech_dataset_images.txt', 'r') as txt:
        paths = tf.convert_to_tensor(txt.read().split('\n'), dtype=tf.string)
    with open('../tf/caltech_dataset_labels.txt', 'r') as txt:
        labels = tf.convert_to_tensor(
            [int(n) for n in txt.read().split('\n')],
            dtype=tf.int32
        )

    train_data = tf.data.Dataset.from_tensor_slices((paths, labels))
    train_data = train_data.take(7378)
    valid_data = tf.data.Dataset.from_tensor_slices((paths, labels))
    valid_data = valid_data.skip(7378).take(513)
    train_data = train_data.map(parser)
    valid_data = valid_data.map(parser)
    train_data = train_data.batch(128)
    valid_data = valid_data.batch(128)

    session = tf.Session()
    model = AlexNet(101)
    model.load_weights(session, 'alexnet_weights.npy')
    # drop_filters(session, model, train_data, valid_data, drop_n=50,
    #              vars_file='../tf/variables/alexnet-caltech101-71')
    model.train(session, train_data, valid_data,
                epochs=80,
                train_layers=['fc8'],
                model_name='alexnet-caltech101')
    session.run(model.iterator.make_initializer(valid_data))
    print('final eval: {}'.format(model.eval(session)))
    session.close()


if __name__ == '__main__':
    main()
