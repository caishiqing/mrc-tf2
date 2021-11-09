from transformers import AutoTokenizer, AutoConfig
from tensorflow.keras import layers
from optimizer import AdamWarmup
from typing import List, Union
from model import build_model
from data import DataLoader
import tensorflow as tf
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
              train_data: Union[List[dict], pd.DataFrame],
              valid_data: Union[List[dict], pd.DataFrame] = None,
              valid_size: float = 0.1,
              model_name: str = None,
              evaluate: bool = True,
              **kwargs):

        if isinstance(train_data, pd.DataFrame):
            train_data = train_data.to_dict(orient='records')
        if valid_data is not None and isinstance(valid_data, pd.DataFrame):
            valid_data = valid_data.to_dict(orient='records')

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

    def test(self, test_data: List[dict], batch_size: int = 32):
        test_examples = self.data_loader.pre_process(test_data)
        x_test, _ = self.data_loader.get_dataset(test_examples, mode='test')
        preds = self.model.predict(x_test, batch_size=batch_size)
        results = self.data_loader.post_process(test_examples, preds)
        return results

    def fix_offset(self, data: Union[List[dict], pd.DataFrame], batch_size: int = 32):
        import re
        from copy import deepcopy

        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')

        ambiguous_records = []
        for record in data:
            context, answer = record['context'], record['answer']
            answer_offsets = [(s.start(), s.end()) for s in re.finditer(re.escape(answer), re.escape(context))]
            if len(answer_offsets) > 1:
                for answer_start, answer_end in answer_offsets:
                    _record = deepcopy(record)
                    _record['answer_start'] = answer_start
                    _record['answer_end'] = answer_end
                    ambiguous_records.append(_record)

        examples = self.data_loader.pre_process(ambiguous_records)
        examples = [example for example in examples if example['label'] != 0]
        inputs, labels = self.data_loader.get_dataset(examples, mode='train')
        preds = self.model.predict(inputs, batch_size=batch_size)
        fixed_offsets = []
        for example, label, pred in zip(examples, labels, preds):
            fixed_offsets.append(
                {
                    'id': example['id'],
                    'score': pred[label],
                    'answer_start': example['answer_start'],
                    'answer_end': example['answer_end']
                }
            )
        fixed_offsets = pd.DataFrame(fixed_offsets).sort_values(
            'score', ascending=False).drop_duplicates(['id']).set_index(
            'id', drop=True).to_dict(orient='index')

        fixed_data = []
        for record in data:
            pid = record['id']
            if pid in fixed_offsets:
                record['answer_start'] = fixed_offsets[pid]['answer_start']
                if 'answer_end' in record:
                    record['answer_end'] = fixed_offsets[pid]['answer_end']

            fixed_data.append(record)

        return fixed_data


if __name__ == '__main__':
    model_path = 'models/webqa'
    data = [
        {
            'id': 'aaa',
            'question': '姚明妻子是谁？',
            'context': '姚明妻子是叶莉，他与叶莉有一个女儿。',
            'answer': '叶莉',
            'answer_start': 10
        },
        {
            'id': 'bbb',
            'question': '姚明有多高？',
            'context': '226cm有多高，姚明身高226cm，被称为小巨人。',
            'answer': '226cm',
            'answer_start': 0
        }

    ]
    pipeline = Pipeline.from_pretrained(model_path)
    fixed_data = pipeline.fix_offset(data)
    print(fixed_data)
