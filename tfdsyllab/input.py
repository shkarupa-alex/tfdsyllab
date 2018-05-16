from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def train_input_fn(wildcard, batch_size):
    # Create dataset from multiple CSV files
    dataset = tf.data.Dataset.list_files(wildcard)

    # Parse CSV record
    def _parse_line(line):
        fields = tf.decode_csv(line, [[''], ['']])
        return {'word': fields[0]}, fields[1]

    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename)
            .map(_parse_line, num_parallel_calls=batch_size),
        cycle_length=5
    )

    # Create batch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 10)

    return dataset


def predict_input_fn(words, batch_size):
    # Create dataset from document list
    dataset = tf.data.Dataset.from_tensors({'word': words})

    # Re-batch input
    dataset = dataset.apply(tf.contrib.data.unbatch())
    dataset = dataset.batch(batch_size)

    return dataset
