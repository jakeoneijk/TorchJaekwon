#type
from typing import List,Union
from torch import Tensor
#package
import os
import torch
import torch.nn as nn
from tqdm import tqdm
#torchjaekwon
from TorchJaekwon.GetModule import GetModule
from TorchJaekwon.DataProcess.Util.UtilData import UtilData
#internal
from HParams import HParams

class Inferencer():
    def __init__(self) -> None:
        self.h_params = HParams()
        self.get_module = GetModule()
        self.util_data = UtilData()

        self.model:Union[nn.Module,object] = GetModule.get_model(self.h_params.model.class_name)
        self.output_dir:str = None
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    
    def get_inference_meta_data_list(self) -> List[dict]:
        meta_data_list = list()
        for data_name in self.h_params.data.data_config_per_dataset_dict:
            meta:list = self.util_data.pickle_load(f'{self.h_params.data.root_path}/{data_name}_test.pkl')
            meta_data_list += meta
        return meta_data_list

    def set_output_dir_path_by_pretrained_name_and_meta_data(self, pretrained_name:str, meta_data:dict) -> None:
        self.output_dir:str = f"{self.h_params.inference.output_dir}/{self.h_params.mode.config_name}({pretrained_name})/{meta_data['name']}"
    
    def read_data_dict_by_meta_data(self,meta_data:dict)->dict:
        '''
        {
            "model_input":
            "gt": {
                "audio",
                "spectrogram"
            }
        }
        '''
        data_dict = dict()
        data_dict["gt"] = dict()
        data_dict["pred"] = dict()
    
    def post_process(self,data_dict:dict)->dict:
        return data_dict

    def save_data(self,data_dict:dict):
        pass
    
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def inference(self) -> None:
        pretrained_path_list:List[str] = self.get_pretrained_path_list()

        for pretrained_path in pretrained_path_list:
            self.pretrained_load(pretrained_path) 
            pretrained_name:str = UtilData.get_file_name_from_path(pretrained_path)
            meta_data_list:List[dict] = self.get_inference_meta_data_list()
            for meta_data in tqdm(meta_data_list,desc='inference by meta data'):
                self.set_output_dir_path_by_pretrained_name_and_meta_data(pretrained_name,meta_data)
                    
                os.makedirs(self.output_dir,exist_ok=True)

                data_dict:dict = self.read_data_dict_by_meta_data(meta_data=meta_data)
                data_dict = self.update_data_dict_by_model_inference(data_dict)
                    
                data_dict:dict = self.post_process(data_dict)
                self.save_data(data_dict)    
    
    def update_data_dict_by_model_inference(self,data_dict):
        if type(data_dict["model_input"]) == Tensor:
            with torch.no_grad():
                data_dict["pred"] = self.model(data_dict["model_input"].to(self.h_params.resource.device))
        return data_dict
                    
    def get_pretrained_path_list(self) -> List[str]:
        pretrain_dir:str = self.h_params.mode.config_name if self.h_params.inference.pretrain_dir == '' else self.h_params.inference.pretrain_dir
        pretrain_dir = f"{self.h_params.inference.pretrain_root_dir}/{pretrain_dir}"
        
        if self.h_params.inference.pretrain_module_name in ["all","last_epoch"]:
            pretrain_name_list:List[str] = [
                pretrain_module
                for pretrain_module in os.listdir(pretrain_dir)
                if pretrain_module.endswith("pth") and "checkpoint" not in pretrain_module
                ]
            pretrain_name_list.sort()

            if self.h_params.inference.pretrain_module_name == "last_epoch":
                pretrain_name_list = [pretrain_name_list[-1]]
        else:
            pretrain_name_list:List[str] = [self.h_params.inference.pretrain_module_name]
        
        return [f"{pretrain_dir}/{pretrain_name}" for pretrain_name in pretrain_name_list]
    
    def pretrained_load(self,pretrain_path:str) -> None:
        if pretrain_path is None:
            return
        pretrained_load:dict = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(pretrained_load)
        self.model = self.model.to(self.h_params.resource.device)
        self.model.eval()