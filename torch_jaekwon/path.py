import os
import torch_jaekwon

TORCH_JAEKWON_PATH:str = os.path.dirname(torch_jaekwon.__file__)

CLASS_DIR_PATH_DICT:dict = {
    'trainer': './train/trainer',
    'lr_scheduler': './train/optimizer/scheduler',
    'loss': './train/loss'
}