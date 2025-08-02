import os
import torch_jaekwon
from dataclasses import dataclass

TORCH_JAEKWON_PATH:str = os.path.dirname(torch_jaekwon.__file__)

CONFIG_DIR = "./config"

@dataclass
class ClassDirs:
    preprocessor:str = './data/preprocess'
    model:str = './model'
    trainer:str = './train/trainer'
    pytorch_dataset:str = './data/dataset'
    lr_scheduler:str = './train/optimizer/scheduler'
    loss:str = './train/loss'
    inferencer:str = './inference'
    evaluator:str = './evaluate/evaluator'
CLASS_DIRS:dict = ClassDirs()

# export ARTIFACTS_ROOT=/path/to/your/output
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts")

@dataclass
class ArtifactsDirs:
    data:str = f'{ARTIFACTS_ROOT}/data'
    preprocessed_data:str = f'{ARTIFACTS_ROOT}/data/preprocessed'
    train:str = f'{ARTIFACTS_ROOT}/train'
    inference:str = f'{ARTIFACTS_ROOT}/inference'
    evaluate:str = f'{ARTIFACTS_ROOT}/evaluate'
ARTIFACTS_DIRS = ArtifactsDirs()

SOURCE_DATA_DIR = os.environ.get('SOURCE_DATA_DIR', f'{ARTIFACTS_ROOT}/data/source')
