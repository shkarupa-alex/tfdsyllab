# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import numpy as np
import os
import re
import sys
import tensorflow as tf
import unicodecsv as csv


def example_from_markup(markup):
    markup = markup.strip()

    # S - Soft border at right (could be wrapped)
    # H - Hard border at right (could not be wrapped)
    # O - Other at right (syllable core, dash, etc.)
    char2label = collections.defaultdict(lambda: 'O', **{'|': 'S', '.': 'H'})
    simple = markup.replace('|.|', '.').replace('|-', '-')
    assert simple.count('||') == 0
    simple = re.sub('.([\.\|])', r'\1', simple)
    labels = u''.join([char2label[c] for c in simple])

    features = markup.replace('.', '').replace('|', '')

    assert len(features) == len(labels), u'{} {}'.format(features, labels)

    return features, labels


def write_csv_dataset(dest_path, set_title, rec_size, set_data):
    dest_mask = os.path.join(dest_path, '{}-{}.csv'.format(set_title, '{}'))
    for i in range(1 + len(set_data) // rec_size):
        rec_data, set_data = set_data[:rec_size], set_data[rec_size:]
        file_name = dest_mask.format(i)

        with open(file_name, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, encoding='utf-8')

            for features, labels in rec_data:
                csvwriter.writerow([features, labels])
            tf.logging.info('Saved {} examples in {}'.format(len(rec_data), file_name))


def main(argv):
    del argv
    tf.logging.info('Loading source dataset from {}'.format(FLAGS.src_file.name))
    syllables = FLAGS.src_file.read().decode('utf-8').split('\n')
    syllables = list(set(syllables))
    np.random.shuffle(syllables)  # shuffle before splitting datasets

    tf.logging.info('Preparing to make datasets')
    examples = [example_from_markup(s) for s in syllables]
    examples_count = len(examples)
    valid_count = int(math.floor(examples_count * FLAGS.valid_size))
    test_count = int(math.floor(examples_count * FLAGS.test_size))
    train_count = examples_count - test_count - valid_count

    try:
        os.makedirs(FLAGS.dest_path)
    except:
        pass

    tf.logging.info('Processing training dataset ({} unique examples)'.format(train_count))
    train_smaples = examples[:train_count]
    write_csv_dataset(FLAGS.dest_path, 'train', FLAGS.rec_size, train_smaples)

    tf.logging.info('Processing validation dataset ({} examples)'.format(valid_count))
    valid_smaples = examples[train_count: train_count + valid_count]
    write_csv_dataset(FLAGS.dest_path, 'valid', FLAGS.rec_size, valid_smaples)

    tf.logging.info('Processing test dataset ({} examples)'.format(test_count))
    test_smaples = examples[train_count + valid_count:]
    write_csv_dataset(FLAGS.dest_path, 'test', FLAGS.rec_size, test_smaples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create training/validation/test datasets')
    parser.add_argument(
        'src_file',
        type=argparse.FileType('rb'),
        help=u'Text file with one word per line. Syllabs should be marked out with wiktionary markup')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Directory where to store TFRecord files')
    parser.add_argument(
        '-rec_size',
        type=int,
        default=5000,
        help='Examples per TFRecord file')
    parser.add_argument(
        '-valid_size',
        type=float,
        default=0.01,
        help='Proportion of examples to include in validation dataset')
    parser.add_argument(
        '-test_size',
        type=float,
        default=0.01,
        help='Proportion of examples to include in test dataset')

    FLAGS, unparsed = parser.parse_known_args()
    assert FLAGS.valid_size + FLAGS.test_size <= 1
    assert not os.path.exists(FLAGS.dest_path) or os.path.isdir(FLAGS.dest_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
