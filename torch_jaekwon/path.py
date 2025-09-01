from typing import Literal
import os
import torch_jaekwon
from dataclasses import dataclass

TORCH_JAEKWON_PATH:str = os.path.dirname(torch_jaekwon.__file__)
CONFIG_DIR = "./config"
ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "./artifacts") # export ARTIFACTS_ROOT=/path/to/your/output
SOURCE_DATA_DIR = os.environ.get('SOURCE_DATA_DIR', f'{ARTIFACTS_ROOT}/data/source')
START_DIR_MAP = {
    'source_data': SOURCE_DATA_DIR,
    'artifacts': ARTIFACTS_ROOT
}

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

@dataclass
class ArtifactsDirs:
    data:str = f'{ARTIFACTS_ROOT}/data'
    preprocessed_data:str = f'{ARTIFACTS_ROOT}/data/preprocessed'
    train:str = f'{ARTIFACTS_ROOT}/train'
    inference:str = f'{ARTIFACTS_ROOT}/inference'
    evaluate:str = f'{ARTIFACTS_ROOT}/evaluate'
ARTIFACTS_DIRS = ArtifactsDirs()

def relpath(
    file_path:str,
    start_dir_type:Literal['source_data', 'artifacts'] = 'source_data'
) -> str:
    file_path_abs:str = os.path.abspath(file_path)
    start_dir_path_abs:str = os.path.abspath(START_DIR_MAP.get(start_dir_type))
    if file_path_abs.startswith(start_dir_path_abs):
        return os.path.relpath(file_path_abs, start=start_dir_path_abs)
    else:
        return None

def abspath(
    file_path:str,
    start_dir_type:Literal['source_data', 'artifacts'] = 'source_data'
) -> str:
    start_dir_path_abs:str = os.path.abspath(START_DIR_MAP.get(start_dir_type))
    abs_path:str = f'{start_dir_path_abs}/{file_path}'
    if os.path.exists(abs_path):
        return abs_path
    else:
        return None
