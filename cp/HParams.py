import yaml
import torch
from dataclasses import dataclass

from TorchJaekwon.Util.UtilData import UtilData

@dataclass
class Mode:
    config_name:str = str()
    config_path:str = f"./Config/{config_name}.yaml"

    stage:str = {0:"preprocess", 1:"train", 2:"inference", 3:"evaluate"}[0]

    train:str = ["start","resume"][0]
    resume_path:str = f"./Train/Log/{config_name}"
    debug_mode:bool = False

@dataclass
class Resource:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    multi_gpu = False
    preprocess = {'num_workers': 20}

@dataclass
class Data:
    original_data_dir:str = "../220101_data"
    root_path:str = "./Data/Dataset"
    config = dict()
    config_per_dataset_dict = dict()

@dataclass
class Logging():
    class_root_dir:str = "./Train/Log"
    project_name:str = "ldm_enhance"
    visualizer_type = ["tensorboard","wandb"][0]
    use_currenttime_on_experiment_name:bool = False
    log_every_local_step:int = 40

@dataclass
class PytorchData:
    class_root_dir:str = "./Data/PytorchDataset"
    dataloader = dict()

@dataclass
class Model:
    class_root_dir:str = "./Model"
    class_name:str = ''
    
@dataclass
class Train:
    class_meta = { 'name' : 'Trainer', 'args' : {}}
    seed_strict = False
    seed = (int)(torch.cuda.initial_seed() / (2**32))
    lr:int = 0.001
    lr_decay:float = 0.98
    lr_decay_step:float = 1.0E+3
    save_model_after_epoch:int = 200
    save_model_every_epoch:int = 100
    check_evalstep_first:bool = True

@dataclass
class Inference():
    class_meta = {'name': 'Inferencer'}
    set_type:str = [ 'single', 'dir', 'testset' ][0]
    set_meta_dict = {
        'single': "./Test/TestInput/commercial_song.wav",
        'dir': ''
    }

    pretrain_module_name:str = ["all","last_epoch"][0]
    pretrain_root_dir:str = "./Train/Log"
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
        self.set_h_params_from_dict(config_dict)
    
    def set_config(self,config_path:str = None) -> None:
        if config_path is not None:
            self.mode.config_name = config_path.split('Config/')[-1]
            self.mode.config_path = config_path
        self.mode.resume_path = f"./Train/Log/{self.mode.config_name}"
        self.load_config()
    
    def set_h_params_from_dict(self, h_params_dict:dict) -> None:
        for data_class_name in h_params_dict:
            for var_name in h_params_dict[data_class_name]:
                setattr(getattr(self,data_class_name),var_name,h_params_dict[data_class_name][var_name])