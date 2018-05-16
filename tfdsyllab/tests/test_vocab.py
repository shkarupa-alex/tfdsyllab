# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import shutil
import tempfile
import unittest
from ..vocab import Vocabulary, extract_vocab


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testFit(self):
        vocab = Vocabulary()
        vocab.fit([b'1', b'2', b'1', b'2', b'3', b'.'])
        self.assertEqual(vocab.items(), [b'1', b'2', b'.', b'3'])
        self.assertEqual(vocab.size(), 4)

    def testFitTrim(self):
        vocab = Vocabulary()
        vocab.fit([b'1', b'2', b'1', b'2', b'3', b'.'])
        vocab.trim(2)
        self.assertEqual(vocab.items(), [b'1', b'2'])

    def testExport(self):
        vocab = Vocabulary()
        vocab.fit([b'1', b'2', b'1', b'2', u'а'.encode('utf-8'), b'.'])

        vocab_filename = os.path.join(self.temp_dir, 'vocab.txt')
        vocab.export(vocab_filename)

        expected = u'1\n2\n.\nа\n'

        with open(vocab_filename, 'rb') as vf:
            result = vf.read().decode('utf-8')
        self.assertEqual(expected, result)


class TestExtractVocab(unittest.TestCase):
    def testNormal(self):
        wildcard = os.path.join(os.path.dirname(__file__), 'data', 'test*.csv')
        vocab = extract_vocab(wildcard)
        vocab.trim(5)

        expected = [u'а', u'и', u'р', u'о', u'т', u'е', u'н', u'в', u'л', u'м', u'ь']
        result = [c.decode('utf-8') for c in vocab.items()]
        self.assertEqual(expected, result)
