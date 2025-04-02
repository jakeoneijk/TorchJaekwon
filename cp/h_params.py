import yaml
import torch
from dataclasses import dataclass

CONFIG_DIR = "./config"
LOG_DIR = "./train/log"

@dataclass
class Mode:
    config_name:str = str()
    config_path:str = f"./{CONFIG_DIR}/{config_name}.yaml"

    stage:str = {0:"preprocess", 1:"train", 2:"inference", 3:"evaluate"}[0]

    train:str = ["start","resume"][0]
    resume_path:str = f"{LOG_DIR}/{config_name}"
    debug_mode:bool = False

@dataclass
class Resource:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multi_gpu = False
    preprocess = {'num_workers': 20}

@dataclass
class Data:
    original_data_dir:str = ""
    config = dict()
    config_per_dataset_dict = dict()

@dataclass
class Logging():
    class_root_dir:str = LOG_DIR
    project_name:str = "ldm_enhance"
    visualizer_type = ["tensorboard","wandb"][0]
    use_currenttime_on_experiment_name:bool = False
    log_step_interval:int = 40

@dataclass
class PytorchData:
    dataloader = dict()

@dataclass
class Model:
    class_name:str = ''
    
@dataclass
class Train:
    class_meta = { 'name' : 'Trainer', 'args' : {}}
    seed_strict = False
    seed = (int)(torch.cuda.initial_seed() / (2**32))
    save_model_after_epoch:int = 200
    save_model_epoch_interval:int = 100
    check_evalstep_first:bool = True

@dataclass
class Inference():
    class_meta = {'name': 'Inferencer'}
    set_type:str = [ 'single', 'dir', 'testset' ][0]
    set_meta_dict = {
        'single': "./Test/TestInput/commercial_song.wav",
        'dir': ''
    }

    ckpt_name:str = ["all","last_epoch"][0]
    pretrain_root_dir:str = LOG_DIR
    pretrain_dir:str = ""
    
    output_dir:str = "./Inference/Output"
    

@dataclass
class Evaluate():
    class_meta = { 'name': 'Evaluater', 'args': {}}
    class_root_dir:str = "./Evaluater"
    source_dir:str = ""

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
        self.pytorch_data = PytorchData()
        self.model = Model()
        self.train= Train()
        self.log = Logging()
        self.inference = Inference()
        self.evaluate = Evaluate()
        self.load_config()
    
    def load_config(self) -> None:
        if self.mode.config_path is None:
            return
        with open(self.mode.config_path, 'r') as yaml_file:
            config_dict:dict = yaml.safe_load(yaml_file)
        self.set_h_params_by_dict(config_dict)
    
    def set_config(self, config_name:str = None) -> None:
        self.mode.config_name = config_name
        self.mode.config_path = f"./{CONFIG_DIR}/{config_name}.yaml"
        self.mode.resume_path = f"{LOG_DIR}/{self.mode.config_name}"
        self.load_config()
    
    def set_h_params_by_dict(self, h_params_dict:dict) -> None:
        for data_class_name in h_params_dict:
            for var_name in h_params_dict[data_class_name]:
                setattr(getattr(self,data_class_name),var_name,h_params_dict[data_class_name][var_name])