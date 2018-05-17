from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

setup(
    name='tfdsyllab',
    version='1.0.0',
    description='Deep word syllables model implemented with Tensorflow',
    url='https://github.com/shkarupa-alex/tfdsyllab',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=['tfdsyllab'],
    install_requires=[
        'tensorflow>=1.5.0',
        'tfunicode>=1.4.4',
        'unicodecsv>=0.14.1',
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)
