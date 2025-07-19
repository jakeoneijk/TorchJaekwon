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

@dataclass
class ArtifactsDirs:
    root:str = './artifacts'
    data:str = f'{root}/data'
    preprocessed_data:str = f'{data}/preprocessed'
    log:str = f'{root}/log'
    inference_output:str = f'{root}/inference_output'
    evaluation_result:str = f'{root}/evaluation_result'
ARTIFACTS_DIRS = ArtifactsDirs()