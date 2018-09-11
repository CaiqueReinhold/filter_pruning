import numpy as np
import tensorflow as tf


def min_max_scaling(arr):
    max_ = arr.max()
    min_ = arr.min()
    return (arr - min_) / (max_ - min_)


def drop_filters(session, model, drop_dataset, valid_dataset, drop_n=50,
                 loss_margin=None, drop_total=50):
    masks = tf.get_collection('MASK')
    conv_ops = tf.get_collection('conv_ops')
    dropped_filters = [set() for _ in range(len(conv_ops))]

    for var in masks:
        mask = np.ones(var.shape)
        session.run(var.assign(mask))

    drop_data_init = model.iterator.make_initializer(drop_dataset)
    valid_data_init = model.iterator.make_initializer(valid_dataset)

    session.run(valid_data_init)
    last_eval = model.eval(session)
    keep = True
    while keep:
        session.run(drop_data_init)

        accumulated = session.run(conv_ops)
        n_batches = 1
        try:
            while True:
                conv_results = session.run(conv_ops)
                accumulated = [
                    (accumulated[i] + conv_results[i])
                    for i in range(len(conv_ops))
                ]
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        accumulated = [
            accumulated[i] / n_batches
            for i in range(len(conv_ops))
        ]

        accumulated = [min_max_scaling(arr) for arr in accumulated]
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
            for i in range(len(conv_ops))
        ]
        for i in range(len(drop)):
            dropped_filters[i].update(drop[i])

        for var, dropped in zip(masks, dropped_filters):
            new_mask = np.ones(var.shape)
            for f in dropped:
                new_mask[:, :, :, f] = 0
            session.run(var.assign(new_mask))

        session.run(valid_data_init)
        new_eval = model.eval(session)

        n_dropped = np.sum([len(s) for s in dropped_filters])

        print('Eval after {} filters dropped: {:.3f}%'.format(
            n_dropped, new_eval * 100
        ))

        if loss_margin is not None:
            if new_eval <= (last_eval * loss_margin):
                keep = False
        else:
            if n_dropped >= drop_total:
                keep = False

    return dropped_filters
