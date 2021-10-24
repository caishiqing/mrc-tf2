from genericpath import exists
from pipeline import Pipeline
import tensorflow as tf
import pandas as pd
import yaml
import fire
import os


def train(train_path: str,
          save_path: str,
          valid_path: str = None,
          evaluate: bool = True,
          gpu: str = '0',
          **kwargs):

    if isinstance(gpu, int):
        gpu = str(gpu)
    elif isinstance(gpu, tuple):
        gpu = ','.join(map(str, gpu))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.makedirs(save_path, exist_ok=True)
    strategy = tf.distribute.MirroredStrategy()
    gpu_num = strategy.num_replicas_in_sync

    params = yaml.load(open('params.yaml'), Loader=yaml.SafeLoader)
    model_params = params['model']
    train_params = params['train']
    train_params.update(kwargs)
    train_params['batch_size'] *= max(1, gpu_num)

    def read_data(path):
        if path is None:
            return None
        if path.endswith('json'):
            df = pd.read_json(path, lines=True)
        elif path.endswith('csv'):
            df = pd.read_csv(path)
        return df.to_dict(orient='records')

    train_data = read_data(train_path)
    valid_data = read_data(valid_path)

    with strategy.scope():
        pipeline = Pipeline(model_params)
        pipeline.train(
            save_path, train_data, valid_data,
            alid_size=train_params.pop('valid_size'),
            evaluate=evaluate, **train_params
        )


if __name__ == '__main__':
    fire.Fire()
