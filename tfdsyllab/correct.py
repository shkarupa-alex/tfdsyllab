# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import re
import sys
import tensorflow as tf
import unicodedata


def vowel_count(word, vowel_letters):
    counter = collections.Counter()
    counter.update(word)

    vowels = 0
    for v in vowel_letters:
        vowels += counter[v]

    return vowels


def simple_correct(example, vowel_letters):
    return len(example) == 1 \
           or len(example) > 0 \
              and example.replace('|.|', '|').strip('-|').count('|') + 1 == vowel_count(example, vowel_letters)


def clean_example(example):
    # Remove accents
    example = u'{}'.format(example)
    example = unicodedata.normalize('NFD', example)
    example = example.replace(u'\u0300', '').replace(u'\u0301', '')
    example = unicodedata.normalize('NFC', example)

    # Remove LTR, RTL marks
    example = example.replace(u'\u200F', '').replace(u'\u200E', '')

    # Replace unusual dash and pause marks
    example = example.replace(u'­', '-').replace(u'‐', '-')
    example = example.replace(u'•', '|.|')

    # Wrap dash and pause marks with syllable marks
    example = example.replace('-', '|-|')
    example = example.replace('.', '|.|')

    # Remove repeated syllables
    example = re.sub('\|+', '|', example)

    return example.strip('-|=!?')


def main(argv):
    del argv
    tf.logging.info('Loading source data from {}'.format(FLAGS.source_file.name))
    source_content = FLAGS.source_file.read().decode('utf-8')
    source_content = source_content.split('\n')

    tf.logging.info('Loading titles data from {}'.format(FLAGS.title_file.name))
    title_content = FLAGS.title_file.read().decode('utf-8').replace('\n', ' \n ')

    tf.logging.info('Checking and correcting examples')
    correct = []
    incorrect = []
    for example in source_content:
        example = clean_example(example)

        if example.count('|') == 0 \
                and '-' in example \
                and ' {} '.format(example) not in title_content \
                and ' {} '.format(example.replace('-', '')) in title_content:
            example = example.replace('-', '|')

        if example.replace('|-|', '-').replace('|.|', '|').count('|') > 1 and '=' not in example:
            correct.append(example)
            continue

        if '-' not in example:
            if simple_correct(example, FLAGS.vowel_letters):
                correct.append(example)
            else:
                incorrect.append(example)
        else:
            sub_examples = example.split('-')
            fully_correct = True
            for sub_examp in sub_examples:
                if not simple_correct(sub_examp, FLAGS.vowel_letters):
                    fully_correct = False
            if fully_correct:
                correct.append(example)
            else:
                incorrect.append(example)

    tf.logging.info('Saving result')
    correct = list(set(correct))
    correct_content = '\n'.join(correct) + '\n'
    FLAGS.correct_file.write(correct_content.encode('utf-8'))

    incorrect = list(set(incorrect))
    incorrect_content = '\n'.join(incorrect) + '\n'
    FLAGS.incorrect_file.write(incorrect_content.encode('utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split source syllables dataset into correct and incorrect russian (!) examples')
    parser.add_argument(
        'source_file',
        type=argparse.FileType('rb'),
        help=u'Source text file with one word per line. Syllabs should be marked out with wiktionary markup')
    parser.add_argument(
        'title_file',
        type=argparse.FileType('rb'),
        help=u'Text file with original article titles')
    parser.add_argument(
        'correct_file',
        type=argparse.FileType('wb'),
        help=u'File to write correct examples')
    parser.add_argument(
        'incorrect_file',
        type=argparse.FileType('wb'),
        help=u'File to write incorrect examples')
    parser.add_argument(
        'vowel_letters',
        nargs='?',
        default=u'аеёиоуыэюя'.encode('utf-8'),
        help=u'Vowel letters. Russian by default')

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.vowel_letters = list(FLAGS.vowel_letters.decode('utf-8'))

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
