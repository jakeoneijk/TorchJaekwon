import sys,os
import yaml
import torch
from dataclasses import dataclass

@dataclass
class Mode:
    config_name:str = "000000_template"
    config_path:str = f"./Config/{config_name}.yaml"

    stage:str = {0:"preprocess", 1:"make_meta_data", 2:"train", 3:"inference", 4:"evaluate"}[0]

    train:str = ["start","resume"][0]
    resume_path:str = f"./Train/Log/{config_name}"
    debug_mode:bool = False

@dataclass
class Logging():
    class_root_dir:str = "./Train/Log"
    project_name:str = "baseline"
    visualizer_type = ["tensorboard","wandb"][1]
    use_currenttime_on_experiment_name:bool = False
    log_every_local_step:int = 40

@dataclass
class Resource:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    multi_gpu = False

@dataclass
class Data:
    original_data_dir:str = "../220101_data"
    root_path:str = "./Data/Dataset"
    data_config_per_dataset_dict = dict()

@dataclass
class PreProcess:
    multi_processing:bool = False
    max_workers:int = None

@dataclass
class MakeMetaData:
    process_dict = {}

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
    class_name:str = 'Trainer'
    seed_strict = False
    seed = (int)(torch.cuda.initial_seed() / (2**32))
    batch_size:int = 32
    lr:int = 0.001
    lr_decay:float = 0.98
    lr_decay_step:float = 1.0E+3
    lr_scheduler_after_valid = False
    lr_scheduler_after_train_step = False
    epoch:int = 3000
    save_model_after_epoch:int = 200
    save_model_every_epoch:int = 100

@dataclass
class Inference():
    class_name:str = 'Inferencer'
    dataset_type:str = ["singledata","testset"][1]
    singledata_path:str ="./Test/TestInput/commercial_song.wav"

    pretrain_module_name:str = ["all","last_epoch"][0]
    pretrain_root_dir:str = "./Train/Log"
    pretrain_dir:str = ""
    
    output_dir:str = "./Inference/Output"
    

@dataclass
class Evaluate():
    class_name:str = 'Evaluater'
    class_root_dir:str = "./Evaluater"
    source_dir:str = ""

class Singleton(object):
  _instance = None
  def __new__(class_, *args, **kwargs):
    if not isinstance(class_._instance, class_):
        class_._instance = object.__new__(class_, *args, **kwargs)
    return class_._instance

class HParams(Singleton):
    def __init__(self) -> None:
        
        self.torch_jaekwon_path = '../000000_TorchJAEKWON'
        if os.path.abspath(self.torch_jaekwon_path) not in sys.path: sys.path.append(os.path.abspath(self.torch_jaekwon_path))
        self.mode = Mode()
        self.resource = Resource()
        self.data = Data()
        self.preprocess = PreProcess()
        self.make_meta_data = MakeMetaData()
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
    
    def set_config(self,config_path:str) -> None:
        self.mode.config_path = config_path
        self.load_config()
    
    def set_h_params_from_dict(self, h_params_dict:dict) -> None:
        for data_class_name in h_params_dict:
            for var_name in h_params_dict[data_class_name]:
                setattr(getattr(self,data_class_name),var_name,h_params_dict[data_class_name][var_name])

