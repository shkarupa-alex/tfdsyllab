from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from operator import itemgetter
from tfunicode import transform_lower_case, transform_normalize_unicode, expand_split_chars
from .input import train_input_fn
import argparse
import math
import os
import sys
import tensorflow as tf


class Vocabulary:
    def __init__(self):
        self._cnt = Counter()

    def fit(self, items):
        assert isinstance(items, list), 'items should be a list'
        self._cnt.update(items)

    def trim(self, min_freq):
        for word in list(self._cnt.keys()):
            if self._cnt[word] < min_freq:
                del self._cnt[word]

    def items(self):
        # Due to different behaviour for items with same counts in Python 2 and 3 we should resort result ourselves
        result = self._cnt.most_common()
        if not len(result):
            return []
        result.sort(key=itemgetter(0))
        result.sort(key=itemgetter(1), reverse=True)
        result, _ = zip(*result)

        return list(result)

    def size(self):
        return len(self._cnt)

    def export(self, filename):
        with open(filename, 'wb') as fout:
            for w in self.items():
                fout.write(w + b'\n')


def extract_vocab(wildcard):
    dataset = train_input_fn(wildcard, 100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    features, _ = next_element
    words = features['word']
    words = transform_normalize_unicode(words, 'NFC')
    words = transform_lower_case(words)

    chars = expand_split_chars(words)
    chars = chars.values

    vocab = Vocabulary()
    with tf.Session() as sess:
        # sess.run(iterator.initializer)
        while True:
            try:
                result = sess.run(chars)
            except tf.errors.OutOfRangeError:
                break
            result = [c for c in result if len(c)]
            vocab.fit(result)
    return vocab


def main(argv):
    del argv

    tf.logging.info('Processing training vocabulary with min freq {}'.format(FLAGS.min_freq))
    wildcard = os.path.join(FLAGS.src_path, 'train-*.csv')
    vocab = extract_vocab(wildcard)
    vocab.trim(FLAGS.min_freq)

    oov_buckets = math.floor(vocab.size() * 0.1)
    tf.logging.info('Recommended oov_buckets: {}'.format(oov_buckets))

    vocab_filename = os.path.join(FLAGS.src_path, 'vocabulary.txt')
    vocab.export(vocab_filename)
    tf.logging.info('Vocabulary stored in {}'.format(vocab_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract vocabulary from training dataset')
    parser.add_argument(
        'src_path',
        type=str,
        help='Path with train CSV files')
    parser.add_argument(
        '-min_freq',
        type=int,
        default=2,
        help='Minimum character frequency to leave it in vocabulary')

    FLAGS, unparsed = parser.parse_known_args()
    assert os.path.exists(FLAGS.src_path)
    assert 0 < FLAGS.min_freq

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
