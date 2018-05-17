# -*- coding: utf-8 -*-
import os
import re
import sys

from tfunicode import transform_normalize_unicode, transform_lower_case, transform_upper_case, expand_split_chars
from tensorflow.contrib.saved_model import get_signature_def_by_key
from tensorflow.python.saved_model import loader
from tensorflow.python.tools import saved_model_utils
import tensorflow as tf


class SyllablesDetector:
    def __init__(self, model_dir):
        # model_dir = os.path.abspath(model_dir)
        assert os.path.exists(model_dir), 'model_dir {} does not exist'.format(model_dir)
        assert os.path.isdir(model_dir), 'model_dir {} is not a directory'.format(model_dir)

        meta_graph_def = saved_model_utils.get_meta_graph_def(model_dir, 'serve')
        self.inputs_tensor_info = get_signature_def_by_key(meta_graph_def, 'predict').inputs
        outputs_tensor_info = get_signature_def_by_key(meta_graph_def, 'predict').outputs
        # Sort to preserve order because we need to go from value to key later.
        self.output_tensor_keys = sorted(outputs_tensor_info.keys())
        self.output_tensor_names = [outputs_tensor_info[tensor_key].name for tensor_key in self.output_tensor_keys]

        self.session = tf.Session(graph=tf.Graph())
        loader.load(self.session, ['serve'], model_dir)

    def __del__(self):
        self.session.close()

    def _predict(self, words):
        output_tensor_values = self.session.run(
            self.output_tensor_names,
            feed_dict={
                self.inputs_tensor_info['input'].name: words
            }
        )

        result = {}
        for i, value in enumerate(output_tensor_values):
            key = self.output_tensor_keys[i]
            result[key] = value

        return result

    def detect(self, words):
        assert isinstance(words, list)

        predictions = self._predict(words)
        assert 'letters' in predictions
        assert 'classes' in predictions

        results = []
        for letters, classes in zip(predictions['letters'], predictions['classes']):
            assert len(letters) == len(classes)

            r = []
            for l, c in zip(letters, classes):
                r.append(l)
                if c == b'S':
                    r.append(b'|')
                if c == b'H':
                    r.append(b'|.|')
            results.append(b''.join(r).decode('utf-8'))

        return results
