import os
import TorchJaekwon

TORCH_JAEKWON_PATH:str = os.path.dirname(TorchJaekwon.__file__)
CLASS_DIR_PATH_DICT:dict = {
    'lr_scheduler': './Train/Optimizer/Scheduler',
    'loss': './Train/Loss'
}