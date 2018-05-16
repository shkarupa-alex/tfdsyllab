# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from ..input import train_input_fn, predict_input_fn


class TestTrainInputFn(tf.test.TestCase):
    def testNormal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', 'test*.csv')
        batch_size = 2

        dataset = train_input_fn(wildcard, batch_size)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        expected_words = tf.convert_to_tensor([u'амфиартроз', u'минимущество'], dtype=tf.string)
        expected_labels = tf.convert_to_tensor(['OSOSOSOOOO', 'OOHHOSOSOOOO'], dtype=tf.string)

        with self.test_session() as sess:
            features, labels = sess.run(features)
            self.assertEqual(type(features), dict)
            self.assertEqual(sorted(features.keys()), ['word'])

            self.assertEqual(expected_words.eval().tolist(), features['word'].tolist())
            self.assertEqual(expected_labels.eval().tolist(), labels.tolist())


class TestPredictInputFn(tf.test.TestCase):
    def testNormal(self):
        source = [u'тест']

        dataset = predict_input_fn(source, 1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        with self.test_session() as sess:
            result = sess.run(features)
            self.assertEqual(type(result), dict)
            self.assertEqual(sorted(result.keys()), ['word'])
            self.assertEqual(len(result['word']), 1)
