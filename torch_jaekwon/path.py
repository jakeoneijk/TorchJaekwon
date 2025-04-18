import os
import torch_jaekwon

class AttrDict(dict):
    def __getattr__(self, name):
        if name in dir(dict): raise AttributeError(f"'{name}' conflicts with built-in dict method.")
        try: return self[name]
        except KeyError: raise AttributeError(f"'SafeAttrDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in dir(dict): raise AttributeError(f"Cannot set attribute '{name}' – conflicts with dict method.")
        self[name] = value

TORCH_JAEKWON_PATH:str = os.path.dirname(torch_jaekwon.__file__)
CONFIG_DIR = "./config"

CLASS_DIRS:dict =  AttrDict({
    'preprocessor': './data/preprocessor',
    'trainer': './train/trainer',
    'pytorch_dataset': './data/dataset',
    'lr_scheduler': './train/optimizer/scheduler',
    'loss': './train/loss'
})

ARTIFACTS_ROOT_NAME = "artifacts"
ARTIFACTS_DIRS:dict = AttrDict({
    'data': f"./{ARTIFACTS_ROOT_NAME}/data",
    'log': f"./{ARTIFACTS_ROOT_NAME}/log",
    'inference_output': f"./{ARTIFACTS_ROOT_NAME}/inference_output",
})