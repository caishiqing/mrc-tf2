from sklearn.model_selection import StratifiedKFold
from data import DataLoader, read_data, write_data
from transformers import AutoTokenizer, AutoConfig
from model import build_model
from pipeline import Pipeline
import tensorflow as tf
import pandas as pd
import random
import yaml
import fire
import os


def train(train_path: str,
          save_path: str,
          valid_path: str = None,
          evaluate: bool = True,
          gpu: str = '0',
          **kwargs):

    # normalize gpu
    if isinstance(gpu, int):
        gpu = str(gpu)
    elif isinstance(gpu, tuple):
        gpu = ','.join(map(str, gpu))

    # set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.makedirs(save_path, exist_ok=True)
    strategy = tf.distribute.MirroredStrategy()
    gpu_num = strategy.num_replicas_in_sync

    # load params
    params = yaml.load(open('params.yaml'), Loader=yaml.SafeLoader)
    model_params = params['model']
    train_params = params['train']
    train_params.update(kwargs)
    train_params['batch_size'] *= max(1, gpu_num)

    train_data = read_data(train_path)
    valid_data = read_data(valid_path) if valid_path is not None else None

    with strategy.scope():
        tokenizer = AutoTokenizer.from_pretrained(model_params['model_name_or_path'])
        data_loader = DataLoader(
            tokenizer,
            max_input_length=model_params['max_input_length'],
            max_answer_length=model_params['max_answer_length'],
            doc_stride=model_params['doc_stride']
        )
        config = AutoConfig.from_pretrained(model_params['model_name_or_path'])
        config.update(model_params)
        model = build_model(config)
        pipeline = Pipeline(data_loader, model, config)
        pipeline.train(
            save_path, train_data, valid_data,
            valid_size=train_params.pop('valid_size'),
            evaluate=evaluate, **train_params
        )


def train_kfold(train_path: str,
                save_path: str,
                valid_path: str = None,
                evaluate: bool = True,
                kfold: int = 5,
                gpu: str = '0',
                **kwargs):

    # normalize gpu
    if isinstance(gpu, int):
        gpu = str(gpu)
    elif isinstance(gpu, tuple):
        gpu = ','.join(map(str, gpu))

    # set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.makedirs(save_path, exist_ok=True)
    strategy = tf.distribute.MirroredStrategy()
    gpu_num = strategy.num_replicas_in_sync

    # load params
    params = yaml.load(open('params.yaml'), Loader=yaml.SafeLoader)
    model_params = params['model']
    train_params = params['train']
    train_params.update(kwargs)
    train_params['batch_size'] *= max(1, gpu_num)

    train_data = read_data(train_path, return_type='df')
    valid_data = read_data(valid_path) if valid_path is not None else None
    if valid_path is not None:
        valid_data = read_data(valid_path, return_type='df')
        train_data = pd.concat([train_data, valid_data])

    train_data["kfold"] = -1
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=2021)
    for k, (t_, v_) in enumerate(kf.split(X=train_data, y=train_data['kfold'])):
        train_data.loc[v_, 'kfold'] = k

    for k in range(kfold):
        # k-fold split and training
        train = train_data[train_data['kfold'] != k]
        valid = train_data[train_data['kfold'] == k]
        tf.keras.backend.clear_session()
        with strategy.scope():
            tokenizer = AutoTokenizer.from_pretrained(model_params['model_name_or_path'])
            data_loader = DataLoader(
                tokenizer,
                max_input_length=model_params['max_input_length'],
                max_answer_length=model_params['max_answer_length'],
                doc_stride=model_params['doc_stride']
            )
            config = AutoConfig.from_pretrained(model_params['model_name_or_path'])
            config.update(model_params)
            model = build_model(config)
            pipeline = Pipeline(data_loader, model, config)
            pipeline.train(
                save_path, train, valid,
                valid_size=train_params.pop('valid_size'),
                model_name='model_{}.h5'.format(k),
                evaluate=evaluate, **train_params
            )


def test(model_path: str,
         data_path: str,
         save_path: str = None,
         batch_size: int = 32):

    pipeline = Pipeline.from_pretrained(model_path)
    test_data = read_data(data_path)
    results = pipeline.test(test_data, batch_size)

    if save_path is not None:
        write_data(results, save_path)

    return results


if __name__ == '__main__':
    fire.Fire()
