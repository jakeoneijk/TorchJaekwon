#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["ARTIFACTS_ROOT"] = "/your/custom/path"

import yaml
import torch
from dataclasses import dataclass, field

from torch_jaekwon.path import CONFIG_DIR, ARTIFACTS_DIRS

@dataclass
class Mode:
    config_name:str = str()
    config_path:str = str() #f"./{CONFIG_DIR}/{config_name}.yaml"

    stage:str = {0:"preprocess", 1:"train", 2:"inference", 3:"evaluate"}[0]

    is_train_resume:bool = False
    train_resume_path:str = f"{ARTIFACTS_DIRS.train}/{config_name}"
    debug_mode:bool = True
    #use_torch_compile:bool = False

@dataclass
class Resource:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers:int = 1

@dataclass
class Data:
    config = dict()
    config_per_dataset_dict = dict()

@dataclass
class Logging:
    project_name:str = "ldm_enhance"
    log_tool:str = ["tensorboard","wandb"][0]
    log_step_interval:int = 40

@dataclass
class DataLoader:
    train = dict()
    valid = dict()
    #test = dict()

@dataclass
class Model:
    class_meta = dict()
    
@dataclass
class Train:
    class_meta = dict()
    seed_strict = False
    seed = (int)(torch.cuda.initial_seed() / (2**32))
    start_logging_epoch:int = 0
    save_model_epoch_interval:int = 100
    check_evalstep_first:bool = True

@dataclass
class Inference():
    class_meta = dict()
    set_type:str = ['single', 'dir', 'testset'][0]
    testdata_path:list = field(default_factory=lambda: [
        "./atrifacts/input/test_song.wav"
    ])
    testdata_dir_path:list = field(default_factory=lambda: [
        "./atrifacts/input/test_dir1",
        "./atrifacts/input/test_dir2"
    ])

    ckpt_name:str = ["all","last"][0]
    pretrain_root_dir_path:str = ARTIFACTS_DIRS.train
    pretrain_dir:str = ""

@dataclass
class Evaluate():
    class_meta = dict()
    eval_dir_path_gt:str = ""
    eval_dir_path_pred:str = ""

class Singleton(object):
  _instance = None
  def __new__(class_, *args, **kwargs):
    if not isinstance(class_._instance, class_):
        class_._instance = object.__new__(class_, *args, **kwargs)
        class_._instance.__first_init__()
    return class_._instance

class HParams(Singleton):
    def __init__(self) -> None:
        pass
    
    def __first_init__(self) -> None:
        self.mode = Mode()
        self.resource = Resource()
        self.data = Data()
        self.dataloader = DataLoader()
        self.model = Model()
        self.train= Train()
        self.log = Logging()
        self.inference = Inference()
        self.evaluate = Evaluate()
        self.load_config()
    
    def load_config(self) -> None:
        if not self.mode.config_path:
            return
        with open(self.mode.config_path, 'r') as yaml_file:
            config_dict:dict = yaml.safe_load(yaml_file)
        self.set_h_params_by_dict(config_dict)
    
    def set_config(self, config_path:str = None) -> None:
        if config_path[0] not in ['/', '.']: config_path = f"./{config_path}"
        assert config_path.startswith(CONFIG_DIR), f"Config path must start with {CONFIG_DIR}"
        assert config_path.endswith('.yaml'), "Config path must end with .yaml"
        self.mode.config_path = config_path
        self.mode.config_name = config_path.split(CONFIG_DIR + '/')[-1].replace('.yaml', '')
        self.mode.train_resume_path = f"{ARTIFACTS_DIRS.train}/{self.mode.config_name}"
        self.load_config()
    
    def set_h_params_by_dict(self, h_params_dict:dict) -> None:
        for data_class_name in h_params_dict:
            for var_name in h_params_dict[data_class_name]:
                setattr(getattr(self,data_class_name),var_name,h_params_dict[data_class_name][var_name])