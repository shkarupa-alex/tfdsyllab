from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.contrib.training import HParams
from tfunicode import transform_normalize_unicode, transform_lower_case, transform_upper_case, expand_split_chars
import tensorflow as tf


def syllab_model_fn(features, labels, mode, params):
    with tf.name_scope('features'):
        input_words = features['word']

        # Normalize unicode words with NFC algorithm
        input_words = transform_normalize_unicode(input_words, 'NFC')

        # Convert words to lower case
        input_words = transform_lower_case(input_words)

        # Split words to letters
        input_chars = expand_split_chars(input_words)

        # Compute padded input mask
        input_masks = tf.SparseTensor(
            indices=input_chars.indices,
            values=tf.ones_like(input_chars.values, dtype=tf.int32),
            dense_shape=input_chars.dense_shape
        )

        # Compute actual length
        batch_size = tf.size(input_words)
        input_lengths = tf.sparse_reduce_sum(input_masks, axis=1)
        max_length = tf.reduce_max(input_lengths)

    with tf.name_scope('embed'):
        # Lookup ids for input chars
        char2id = tf.contrib.lookup.index_table_from_tensor(
            params.vocab_items,
            num_oov_buckets=params.oov_buckets
        )
        char_ids = tf.SparseTensor(
            indices=input_chars.indices,
            values=char2id.lookup(input_chars.values),
            dense_shape=input_chars.dense_shape
        )
        char_ids = tf.sparse_tensor_to_dense(char_ids)
        char_ids = tf.reshape(char_ids, [batch_size, max_length])  # required for model export

        # Prepare embedding matrix
        vocab_size = len(params.vocab_items) + params.oov_buckets
        vocab_embeddings = tf.get_variable(
            'char_embeddings',
            [vocab_size, params.embed_size],
            dtype=None,
            initializer=tf.random_uniform_initializer(-1, 1),
        )
        if mode == tf.estimator.ModeKeys.TRAIN and params.embed_dropout > 0:
            vocab_embeddings = tf.nn.dropout(
                vocab_embeddings,
                keep_prob=1 - params.embed_dropout,
                noise_shape=[vocab_size, 1]
            )

        # Embed input char ids
        char_embeddings = tf.nn.embedding_lookup(
            vocab_embeddings,
            char_ids,
        )

    with tf.name_scope('rnn'):
        # Add RNN layer
        cells_fw = [tf.contrib.rnn.LSTMBlockCell(params.rnn_size) for _ in range(params.rnn_layers)]
        cells_bw = [tf.contrib.rnn.LSTMBlockCell(params.rnn_size) for _ in range(params.rnn_layers)]
        if mode == tf.estimator.ModeKeys.TRAIN and params.rnn_dropout > 0:
            cells_fw = [
                tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1 - params.rnn_dropout) for cell in cells_fw]
            cells_bw = [
                tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1 - params.rnn_dropout) for cell in cells_bw]

        rnn_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=char_embeddings,
            sequence_length=input_lengths,
            dtype=tf.float32
        )

    with tf.name_scope('dense'):
        # Add fully connected layer
        rnn_flat = tf.reshape(rnn_outputs, [
            -1,
            params.rnn_size * 2  # due to bidirectional RNN
        ])
        dense_logits = tf.layers.dense(
            inputs=rnn_flat,
            units=3,  # number of classes
            # activation=None  # or with activation?
        )
        dense_logits = tf.reshape(dense_logits, [
            batch_size,
            max_length,
            3  # number of classes
        ])

    with tf.name_scope('lookup'):
        # Label transformation tables
        label_table = tf.constant(['O', 'S', 'H'])
        id2label = tf.contrib.lookup.index_to_string_table_from_tensor(label_table, default_value='O')
        label2id = tf.contrib.lookup.index_table_from_tensor(label_table, default_value=0)

    # Build EstimatorSpec's
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'letters': tf.sparse_tensor_to_dense(
                input_chars,
                default_value=''
            ),
            'classes': id2label.lookup(
                tf.argmax(
                    dense_logits,
                    axis=2
                ),
            ),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            }
        )

    with tf.name_scope('labels'):
        # Transform labels to IDs
        input_labels = transform_upper_case(labels)
        input_labels = expand_split_chars(input_labels)

        label_ids = tf.SparseTensor(
            indices=input_labels.indices,
            values=label2id.lookup(input_labels.values),
            dense_shape=input_labels.dense_shape
        )
        label_ids = tf.sparse_tensor_to_dense(label_ids)

    # Add loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=label_ids,
        logits=dense_logits,
        weights=tf.sparse_tensor_to_dense(input_masks),  # 0 for padded tokens, should reduce padded examples loss to 0
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    )

    # Add metrics
    accuracy_metric = tf.metrics.accuracy(
        labels=label_ids,
        predictions=tf.argmax(dense_logits, axis=2),
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': accuracy_metric}
        )

    # Add optimizer
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer='Adam',
    )

    assert mode == tf.estimator.ModeKeys.TRAIN
    metrics_hook = tf.train.LoggingTensorHook({
        'accuracy': accuracy_metric[1],
    }, every_n_iter=100)

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
        training_hooks=[metrics_hook]
    )


class SyllablesEstimator(estimator.Estimator):
    def __init__(self,
                 vocab_items,  # vocabulary characters
                 oov_buckets,  # number of out of vocabulary char buckets
                 embed_size,  # size of char embedding
                 embed_dropout, # embedding dropout probability
                 rnn_size,  # size of single RNN layer
                 rnn_layers,  # number of RNN layers
                 rnn_dropout,  # 1 - dropout probability
                 learning_rate,  # learning rate
                 model_dir=None,
                 config=None):
        params = HParams(
            vocab_items=vocab_items,
            oov_buckets=oov_buckets,
            embed_size=embed_size,
            embed_dropout=embed_dropout,
            rnn_size=rnn_size,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            learning_rate=learning_rate,
        )

        super(SyllablesEstimator, self).__init__(
            model_fn=syllab_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
        )
