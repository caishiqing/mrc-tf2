from sklearn.model_selection import KFold
from data import DataLoader, read_data, write_data
from transformers import AutoTokenizer, AutoConfig
from typing import Union, List
from model import build_model
from pipeline import Pipeline
import tensorflow as tf
from copy import copy
import pandas as pd
import yaml
import fire
import os


def train(train_path: Union[str, List[dict], pd.DataFrame],
          save_path: Union[str, List[dict], pd.DataFrame],
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
    model_params = copy(params['model'])
    train_params = copy(params['train'])
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


def train_kfold(data: Union[str, List[dict], pd.DataFrame],
                save_path: str,
                evaluate: bool = True,
                kfold: int = 5,
                gpu: str = '0',
                **kwargs):
    from multiprocessing import Process
    data = read_data(data, return_type='df')

    if 'kfold' not in data:
        data["kfold"] = -1
        kf = KFold(n_splits=kfold, shuffle=True, random_state=2021)
        for k, (t_, v_) in enumerate(kf.split(X=data)):
            data.loc[v_, 'kfold'] = k

    for k in range(kfold):
        # k-fold split and training
        train_data = data[data['kfold'] != k]
        valid_data = data[data['kfold'] == k]

        model_name = 'model_{}.h5'.format(k)
        kwargs['valid_path'] = valid_data
        kwargs['evaluate'] = evaluate
        kwargs['model_name'] = model_name
        kwargs['gpu'] = gpu

        process = Process(
            target=train,
            args=(train_data, save_path),
            kwargs=kwargs,
        )
        process.start()
        process.join()


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
