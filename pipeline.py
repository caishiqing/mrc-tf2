from tensorflow.keras import callbacks
from transformers import AutoTokenizer
from optimizer import AdamWarmup
from model import build_model
from data import DataLoader
import tensorflow as tf
from typing import List
from rouge import Rouge
from copy import copy
import pandas as pd
import random
import json
import os


class Pipeline(object):

    def __init__(self, params: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(params['model_name_or_path'])
        self.data_loader = DataLoader(
            self.tokenizer,
            max_input_length=params['max_input_length'],
            max_answer_length=params['max_answer_length'],
            doc_stride=params['doc_stride']
        )
        self.model, self.bert = build_model(
            model_name_or_path=params['model_name_or_path'],
            max_input_length=params['max_input_length'],
            max_answer_length=params['max_answer_length']
        )
        self.params = params

    @classmethod
    def from_pretrained(cls, model_path: str):
        param_path = os.path.join(model_path, 'params.json')
        params = json.load(open(param_path, encoding='utf-8'))
        params['model_name_or_path'] = model_path
        pipeline = cls(params)
        pipeline.model.load_weights(os.path.join(model_path, 'model.h5'))
        return pipeline

    def train(self,
              save_path: str,
              train_data: List[dict],
              valid_data: List[dict] = None,
              valid_size: float = 0.1,
              evaluate: bool = True,
              **kwargs):

        # training params
        batch_size = kwargs.get('batch_size', 8)
        epochs = kwargs.get('epochs', 10)
        steps_per_epoch = kwargs.get('steps_per_epochs', 200)
        learning_rate = kwargs.get('learning_rate', 2e-5)
        warmup_proportion = kwargs.get('warmup_proportion', 0.1)

        # save other files
        params = copy(self.params)
        self.bert.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, 'params.json'), 'w', encoding='utf-8') as fp:
            json.dump(params, fp)

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
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_path, 'model.h5'),
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
            self.model.load_weights(os.path.join(save_path, 'model.h5'))
            preds = self.model.predict(x_valid, batch_size=batch_size)
            results = self.data_loader.post_process(valid_examples, preds)
            valid_df = pd.DataFrame(valid_data)
            result_df = pd.DataFrame(results)
            df = pd.merge(valid_df, result_df, on='id')

            rouge = Rouge()

            def rouge_l(row):
                prediction = self.tokenizer.tokenize(row['prediction'])
                ground_trues = self.tokenizer.tokenize(row['answer'])
                scores = rouge.get_scores(prediction, ground_trues)
                return scores[0]['rouge-l']['f']

            df['rouge'] = df.apply(rouge_l, axis=1)
            score = df['rouge'].mean()
            print('Rouge-L: {:.4}'.format(score))
