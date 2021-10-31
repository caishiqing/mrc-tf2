from transformers import TFAutoModel, AutoConfig
from tensorflow.keras import layers
import tensorflow as tf
from typing import List


class Pointer2D(layers.Layer):
    """2D pointor network"""

    def __init__(self,
                 max_input_length: int = 512,
                 max_answer_length: int = 8,
                 **kwargs):
        """[summary]

        Args:
            max_input_length (int, optional): Max input length
            max_answer_length (int, optional): Max answer length
        """
        super(Pointer2D, self).__init__(**kwargs)
        self.max_input_length = max_input_length
        self.max_answer_length = max_answer_length

        # 2D permutation indicate matrix
        permut_indicate = tf.ones((max_input_length, max_input_length))
        permut_indicate = tf.linalg.band_part(
            permut_indicate, 0, max_answer_length - 1)
        self.indices = tf.where(permut_indicate == 1)

        self.fc = layers.Dense(1)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        embeddings, token_type_ids, attention_mask = inputs
        mask = tf.cast(token_type_ids, embeddings.dtype) *\
            tf.cast(attention_mask, embeddings.dtype)
        start_indices = self.indices[:, 0]
        end_indices = self.indices[:, 1]

        # 2D permutation representation
        start, end = tf.split(embeddings, 2, axis=-1)
        _start = tf.gather(start, start_indices, axis=1)
        _end = tf.gather(end, end_indices, axis=1)
        states = _start + _end

        logits = self.fc(states)
        logits = tf.squeeze(logits, -1)
        _start_mask = tf.gather(mask, start_indices, axis=1)
        _end_mask = tf.gather(mask, end_indices, axis=1)
        _mask = _start_mask * _end_mask
        logits -= 1e7 * (1 - _mask)
        output = tf.nn.softmax(logits, axis=-1)

        return output

    def get_config(self):
        config = super(Pointer2D, self).get_config()
        config.update(
            {
                "max_input_length": self.max_input_length,
                "max_answer_length": self.max_answer_length,
            }
        )
        return config


def build_model(config: AutoConfig) -> tf.keras.Model:

    if hasattr(config, 'model_name_or_path'):
        bert = TFAutoModel.from_pretrained(config.model_name_or_path)
    else:
        bert = TFAutoModel.from_config(config)

    input_ids = layers.Input(shape=(config.max_input_length,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(config.max_input_length,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(config.max_input_length,), dtype=tf.bool)

    embeddings = bert(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    output = Pointer2D(config.max_input_length, config.max_answer_length)(
        [embeddings, token_type_ids, attention_mask]
    )

    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)
    return model
