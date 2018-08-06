import numpy as np
import tensorflow as tf


def min_max_scaling(arr):
    max_ = arr.max()
    min_ = arr.min()
    return (arr - min_) / (max_ - min_)


def drop_filters(model, drop_dataset, valid_dataset, drop_n=100,
                 loss_margin=0.98):
    session = tf.Session()
    saver = tf.train.Saver(tf.get_collection('CONV') +
                           tf.get_collection('FC'))

    masks = tf.get_collection('MASK')
    conv_ops = tf.get_collection('conv_ops')
    dropped_filters = [set() for _ in range(len(model['ops']))]

    for var in masks:
        mask = np.ones(var.shape)
        session.run(var.assign(mask))

    drop_data_init = self.iterator.make_initializer(drop_dataset)
    valid_data_init = self.iterator.make_initializer(valid_dataset)

    session.run(valid_data_init)
    last_eval = model.eval(session)
    keep = True
    while keep:
        session.run(drop_data_init)
        
        accumulated = session.run(conv_ops)
        try:
            while True:
                conv_results = session.run(conv_ops)
                accumulated = [
                    (accumulated[i] + conv_results[i]) / 2
                    for i in range(len(conv_ops))
                ]
        except tf.errors.OutOfRangeError:
            pass

        accumulated = map(min_max_scaling, accumulated)
        summations = [np.zeros((op.shape[3],)) for op in conv_ops]

        for i in range(len(summations)):
            for j in range(summations[i].shape[0]):
                summations[i][j] = (
                    np.mean(accumulated[i][:, :, :, j])
                )

        for i in range(len(summations)):
            for f in dropped_filters[i]:
                summations[i][f] = np.inf

        indexes = [np.argsort(sums)[:drop_n] for sums in summations]
        all_sums = np.concatenate([
            summations[i][indexes[i]] for i in range(len(indexes))
        ])
        all_sums = np.argsort(all_sums)[:drop_n]
        filter_indexes = np.concatenate(indexes)[all_sums]
        drop = [
            filter_indexes[((i * drop_n) <= all_sums) &
                           (all_sums < ((i + 1)) * drop_n)]
            for i in range(len(ops))
        ]
        for i in range(len(drop)):
            dropped_filters[i].update(drop[i])

        for var, dropped in zip(masks, dropped_filters):
            new_mask = np.ones(var.shape)
            for f in dropped:
                new_mask[:, :, :, f] = 0
            session.run(var.assign(new_mask))

        new_eval = model.eval(session)

        print('Eval after {} filters dropped: {:.3f}%'.format(
            np.sum([len(s) for s in dropped_filters]), new_eval * 100
        ))

        if new_eval <= (last_eval * loss_margin):
            keep = False
        else:
            with open('drop_filters.pickle', 'wb') as f:
                pickle.dump(dropped_filters, f)
    session.close()
