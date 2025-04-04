import os
import torch_jaekwon

TORCH_JAEKWON_PATH:str = os.path.dirname(torch_jaekwon.__file__)

CLASS_DIR_PATH_DICT:dict = {
    'preprocessor': './data_process/preprocess',
    'trainer': './train/trainer',
    'data_loader': './data/pytorch_dataloader',
    'pytorch_dataset': './data/pytorch_dataset',
    'lr_scheduler': './train/optimizer/scheduler',
    'loss': './train/loss'
}