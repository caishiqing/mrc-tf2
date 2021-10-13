from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class Pointer2D(layers.Layer):
    """ 2D pointor network """

    def __init__(self, max_sequence_length=512, max_answer_length=8, **kwargs):
        super(Pointer2D, self).__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.max_answer_length = max_answer_length

        # 2D permutation indicate matrix
        permut_indicate = tf.ones((max_sequence_length, max_sequence_length))
        permut_indicate = tf.linalg.band_part(permut_indicate, 0, max_answer_length-1)
        self.indices = tf.where(permut_indicate == 1)

        self.fc = layers.Dense(1)

    def call(self, inputs):
        embeddings, token_type_ids, attention_mask = inputs
        mask = tf.cast(token_type_ids, embeddings.dtype) * tf.cast(attention_mask, embeddings.dtype)
        start_indices = self.indices[:, 0]
        end_indices = self.indices[:, 1],

        # 2D permutation representation
        start, end = tf.split(embeddings, 2, axis=-1)
        _start = tf.gather(start, start_indices, axis=1)
        _end = tf.gather(end,  end_indices, axis=1)
        states = _start + _end

        logits = self.fc(states)
        logits = tf.squeeze(logits, -1)
        _start_mask = tf.gather(mask, start_indices)
        _end_mask = tf.gather(mask, end_indices)
        _mask = _start_mask * _end_mask
        logits -= 1e7 * (1 - _mask)
        output = tf.nn.softmax(logits, axis=-1)

        return output

    def get_config(self):
        config = super(Pointer2D, self).get_config()
        config.update(
            {
                'max_sequence_length': self.max_sequence_length,
                'max_answer_length': self.max_answer_length
            }
        )
        return config


if __name__ == '__main__':
    x = tf.ones((1, 10, 4))
    a, b = tf.split(x, 2, axis=-1)
    print(a.shape, b.shape)
