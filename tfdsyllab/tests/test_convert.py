# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import tensorflow as tf
import unittest
from ..convert import example_from_markup, write_csv_dataset
from ..input import train_input_fn


class TestExampleFromMarkup(unittest.TestCase):
    def testEmpty(self):
        features, labels = example_from_markup('')
        self.assertEqual('', features)
        self.assertEqual('', labels)

    def testUnicode(self):
        features, labels = example_from_markup(u'я|.|по|но|-|рос|сий|ский')
        self.assertEqual(u'японо-российский', features)
        self.assertEqual('HOSOOSOOSOOSOOOO', labels)


class TestWriteCsvDataset(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = [
            (u'японо-российский', 'HOSOOSOOSOOSOOOO')
        ]
        write_csv_dataset(self.temp_dir, 'test', 100, source)

        expected_filename = os.path.join(self.temp_dir, 'test-0.csv')
        dataset = train_input_fn(expected_filename, 1)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with self.test_session() as sess:
            features, labels = sess.run(next_element)
            self.assertEqual(source[0][0].encode('utf-8'), features['word'])
            self.assertEqual(source[0][1].encode('utf-8'), labels)
