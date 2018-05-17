# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model import SyllablesEstimator
from .input import train_input_fn, predict_input_fn, serve_input_fn
import argparse
import os
import sys
import tensorflow as tf


def main(argv):
    del argv

    vocab_name = os.path.join(FLAGS.data_path, 'vocabulary.txt')
    with open(vocab_name, 'rb') as vocab_file:
        vocab_items = vocab_file.read().strip().split(b'\n')

    estimator = SyllablesEstimator(
        vocab_items=vocab_items,
        oov_buckets=FLAGS.oov_buckets,
        embed_size=FLAGS.embed_size,
        embed_dropout=FLAGS.embed_dropout,
        rnn_size=FLAGS.rnn_size,
        rnn_layers=FLAGS.rnn_layers,
        rnn_dropout=FLAGS.rnn_dropout,
        learning_rate=FLAGS.learning_rate,
        model_dir=FLAGS.model_path,
    )

    # Run training
    train_wildcard = os.path.join(FLAGS.data_path, 'train*.csv')
    estimator.train(input_fn=lambda: train_input_fn(train_wildcard, FLAGS.batch_size, FLAGS.num_repeats, True))

    # Run evaluation
    eval_wildcard = os.path.join(FLAGS.data_path, 'test*.csv')
    metrics = estimator.evaluate(input_fn=lambda: train_input_fn(eval_wildcard, FLAGS.batch_size))
    print(metrics)

    if len(FLAGS.export_path):
        estimator.export_savedmodel(FLAGS.export_path, serve_input_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate syllables detection model')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path with train/eval data and vocabulary')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to store model')
    parser.add_argument(
        '-export_path',
        type=str,
        default='',
        help='Path to store model')
    parser.add_argument(
        '-oov_buckets',
        type=int,
        default=5,
        help='Number of OOV char buckets')
    parser.add_argument(
        '-embed_size',
        type=int,
        default=10,
        help='Ngram embedding size')
    parser.add_argument(
        '-embed_dropout',
        type=float,
        default=0.01,
        help='Input char emmbedding dropout probability')
    parser.add_argument(
        '-rnn_size',
        type=int,
        default=7,
        help='RNN layer size')
    parser.add_argument(
        '-rnn_layers',
        type=int,
        default=1,
        help='RNN layers count')
    parser.add_argument(
        '-rnn_dropout',
        type=float,
        default=0.01,
        help='RNN dropout probability')
    parser.add_argument(
        '-learning_rate',
        type=float,
        default=0.005,
        help='Learning rate')
    parser.add_argument(
        '-batch_size',
        type=int,
        default=100,
        help='Examples per batch')
    parser.add_argument(
        '-num_repeats',
        type=int,
        default=10,
        help='Train/eval iterations number')

    FLAGS, unparsed = parser.parse_known_args()
    assert os.path.exists(FLAGS.data_path) and os.path.isdir(FLAGS.data_path)
    assert not os.path.exists(FLAGS.export_path) or os.path.isdir(FLAGS.export_path)
    assert not os.path.exists(FLAGS.model_path) or os.path.isdir(FLAGS.model_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
