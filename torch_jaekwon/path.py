import os
import torch_jaekwon

TORCH_JAEKWON_PATH:str = os.path.dirname(torch_jaekwon.__file__)
CONFIG_DIR = "./config"

CLASS_DIR_PATH_DICT:dict = {
    'preprocessor': './data_process/preprocess',
    'trainer': './train/trainer',
    'data_loader': './data/pytorch_dataloader',
    'pytorch_dataset': './data/pytorch_dataset',
    'lr_scheduler': './train/optimizer/scheduler',
    'loss': './train/loss'
}

ARTIFACTS_DIR = "./artifacts"
ARTIFACTS_DIR_PATH_DICT:dict = {
    'data': f"{ARTIFACTS_DIR}/data",
    'log': f"{ARTIFACTS_DIR}/log",
    'inference_output': f"{ARTIFACTS_DIR}/inference_output",
}