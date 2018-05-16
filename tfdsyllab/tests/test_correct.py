# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from ..correct import vowel_count, simple_correct, clean_example

VOWEL_LETTERS = list(u'аеёиоуыэюя')


class TestVowelCount(unittest.TestCase):
    def testEmpty(self):
        self.assertEqual(0, vowel_count('', VOWEL_LETTERS))

    def testSingleNonVowel(self):
        self.assertEqual(0, vowel_count(u'м', VOWEL_LETTERS))

    def testSingleVowel(self):
        self.assertEqual(1, vowel_count(u'а', VOWEL_LETTERS))

    def testManyVowels(self):
        self.assertEqual(3, vowel_count(u'тестовый', VOWEL_LETTERS))


class TestSimpleCorrect(unittest.TestCase):
    def testEmpty(self):
        self.assertEqual(False, simple_correct('', VOWEL_LETTERS))

    def testSingleVowel(self):
        self.assertEqual(True, simple_correct(u'м', VOWEL_LETTERS))

    def testSingleNonVowel(self):
        self.assertEqual(True, simple_correct(u'а', VOWEL_LETTERS))

    def testIncorrect(self):
        self.assertEqual(False, simple_correct(u'тесты', VOWEL_LETTERS))
        self.assertEqual(False, simple_correct(u'теест', VOWEL_LETTERS))


class TestCleanExample(unittest.TestCase):
    def testEmpty(self):
        self.assertEqual('', clean_example(''))

    def testCorrect(self):
        self.assertEqual(u'м|-|.|об|раз|ный', clean_example(u'м|-|.|об|раз|ный'))

    def testAccents(self):
        self.assertEqual(u'ев|ро|дол|лар', clean_example(u'ѐв|ро|дол|лар'))

    def testWrongChars(self):
        self.assertEqual(u'ев|ро|скеп|ти|цизм', clean_example(u'­ев|ро|скеп|ти|цизм‐'))
        self.assertEqual(u'ме|ан|дри|ро|ва|ни|.|е', clean_example(u'ме|ан|дри|ро|ва|ни•е'))

    def testHalfSyllab(self):
        self.assertEqual(u's|-|.|об|раз|ный', clean_example(u's-|.|об|раз|ный'))
        self.assertEqual(u'у|.|до|сто|ве|рять', clean_example(u'у.|до|сто|ве|рять'))

    def testNoSyllab(self):
        self.assertEqual(u'все|мир|но|-|ис|то|ри|чес|кий', clean_example(u'все|мѝр|но-ис|то|ри|чес|кий'))
        self.assertEqual(u'а|.|на|ло|гий', clean_example(u'а.на|ло|гий'))
