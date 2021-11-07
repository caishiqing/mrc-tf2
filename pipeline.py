from transformers import AutoTokenizer, AutoConfig
from tensorflow.keras import layers
from optimizer import AdamWarmup
from model import build_model
from data import DataLoader
import tensorflow as tf
from typing import List
from rouge import Rouge
from copy import copy
import pandas as pd
import random
import os


class Pipeline(object):

    def __init__(self, data_loader: DataLoader, model: tf.keras.Model, config: AutoConfig):
        self.data_loader = data_loader
        self.tokenizer = data_loader.tokenizer
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, model_path: str):
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        data_loader = DataLoader(
            tokenizer,
            max_input_length=config.max_input_length,
            max_answer_length=config.max_answer_length,
            doc_stride=config.doc_stride
        )

        if hasattr(config, 'model_name_or_path'):
            del config.model_name_or_path

        models = []
        for i, file in enumerate(filter(lambda x: x.endswith('.h5'), os.listdir(model_path))):
            model = build_model(config)
            model._name = '{}_{}'.format(model._name, i)
            model.load_weights(os.path.join(model_path, file))
            models.append(model)

        if len(models) == 0:
            raise Exception('No model weights file in the path!')
        elif len(models) == 1:
            model = models[0]
        else:
            # If trained k models by k-fold
            input_ids = layers.Input(shape=(config.max_input_length,), dtype=tf.int32)
            token_type_ids = layers.Input(shape=(config.max_input_length,), dtype=tf.int32)
            attention_mask = layers.Input(shape=(config.max_input_length,), dtype=tf.bool)
            outputs = []
            for model in models:
                outputs.append(model([input_ids, token_type_ids, attention_mask]))

            # merge models by average
            output = layers.Average(outputs)
            model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)

        pipeline = Pipeline(data_loader, model, config)
        return pipeline

    def train(self,
              save_path: str,
              train_data: List[dict],
              valid_data: List[dict] = None,
              valid_size: float = 0.1,
              model_name: str = None,
              evaluate: bool = True,
              **kwargs):

        # training params
        batch_size = kwargs.get('batch_size', 8)
        epochs = kwargs.get('epochs', 10)
        steps_per_epoch = kwargs.get('steps_per_epoch', 200)
        learning_rate = kwargs.get('learning_rate', 2e-5)
        warmup_proportion = kwargs.get('warmup_proportion', 0.1)

        # save other files
        self.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # get dataset
        random.shuffle(train_data)
        if valid_data is None:
            totle_num = len(train_data)
            valid_num = int(totle_num*valid_size)
            valid_data = train_data[:valid_num]
            train_data = train_data[valid_num:]

        train_examples = self.data_loader.pre_process(train_data)
        valid_examples = self.data_loader.pre_process(valid_data)

        x_train, y_train = self.data_loader.get_dataset(train_examples, 'train')
        x_valid, y_valid = self.data_loader.get_dataset(valid_examples, 'train')

        # build optimizer
        totle_steps = len(y_train)
        warmup_steps = int(totle_steps*warmup_proportion)
        decay_steps = totle_steps-warmup_steps
        optimizer = AdamWarmup(warmup_steps, decay_steps, learning_rate)

        # compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['acc']
        )
        self.model.summary()

        # build checkpoint callback
        model_name = model_name if model_name else 'model.h5'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_path, model_name),
            save_best_only=True,
            save_weights_only=True,
        )

        # train model
        self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(x_valid, y_valid),
            callbacks=[checkpoint]
        )

        if evaluate:
            self.model.load_weights(os.path.join(save_path, model_name))
            preds = self.model.predict(x_valid, batch_size=batch_size)
            results = self.data_loader.post_process(valid_examples, preds)
            valid_df = pd.DataFrame(valid_data)
            result_df = pd.DataFrame(results)
            df = pd.merge(valid_df, result_df, on='id')

            rouge = Rouge()

            def rouge_l(row):
                predictions = ' '.join(self.tokenizer.tokenize(row['prediction']))
                ground_truth = ' '.join(self.tokenizer.tokenize(row['answer']))
                score = rouge.get_scores(predictions, ground_truth)[0]['rouge-l']['f']
                return score

            df['score'] = df.apply(rouge_l, axis=1)
            score = df['score'].mean()
            print('Rouge-L: {:.4}'.format(score))

    def test(self, test_data: List[dict], batch_size=32):
        test_examples = self.data_loader.pre_process(test_data)
        x_test, _ = self.data_loader.get_dataset(test_examples, mode='test')
        preds = self.model.predict(x_test, batch_size=batch_size)
        results = self.data_loader.post_process(test_examples, preds)
        return results
