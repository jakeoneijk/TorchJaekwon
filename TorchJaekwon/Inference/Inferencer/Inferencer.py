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
from TorchJaekwon.Util.UtilData import UtilData 
#internal
from HParams import HParams

class Inferencer():
    def __init__(self,
                 output_dir:str = HParams().inference.output_dir,
                 experiment_name:str = HParams().mode.config_name,
                 model:Union[nn.Module,object] =  None,
                 device:torch.device = HParams().resource.device
                 ) -> None:
        self.output_dir:str = output_dir
        self.experiment_name:str = experiment_name

        self.get_module = GetModule()

        self.device:torch.device = device

        self.model:Union[nn.Module,object] = self.get_model() if model is None else model
    
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

    def get_output_dir_path(self, pretrained_name:str, test_name:str) -> None:
        return f"{self.output_dir}/{self.experiment_name}({pretrained_name})/{test_name}"
    
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

    def save_data(self,output_dir_path,meta_data,data_dict)->None:
        pass
    
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def get_model(self) -> nn.Module:
        return GetModule.get_model(HParams().model.class_name) if (HParams().model.class_name not in [None,'']) else None

    def inference(self,
                  pretrained_root_dir:str = HParams().inference.pretrain_root_dir,
                  pretrained_dir_name:str = HParams().mode.config_name if HParams().inference.pretrain_dir == '' else HParams().inference.pretrain_dir,
                  pretrain_module_name:str = HParams().inference.pretrain_module_name
                  ) -> None:
        pretrained_path_list:List[str] = self.get_pretrained_path_list(
            pretrain_root_dir= pretrained_root_dir,
            pretrain_dir_name = pretrained_dir_name,
            pretrain_module_name= pretrain_module_name
        )

        for pretrained_path in pretrained_path_list:
            self.pretrained_load(pretrained_path) 
            pretrained_name:str = UtilData.get_file_name_from_path(pretrained_path)
            meta_data_list:List[dict] = self.get_inference_meta_data_list()
            for meta_data in tqdm(meta_data_list,desc='inference by meta data'):
                output_dir_path:str = self.get_output_dir_path(pretrained_name=pretrained_name,test_name=meta_data["test_name"])

                data_dict:dict = self.read_data_dict_by_meta_data(meta_data=meta_data)
                data_dict = self.update_data_dict_by_model_inference(data_dict)
                    
                data_dict:dict = self.post_process(data_dict)
                self.save_data(output_dir_path,meta_data,data_dict)    
    
    def update_data_dict_by_model_inference(self,data_dict):
        if type(data_dict["model_input"]) == Tensor:
            with torch.no_grad():
                data_dict["pred"] = self.model(data_dict["model_input"].to(self.h_params.resource.device))
        return data_dict
                    
    def get_pretrained_path_list(self,
                                 pretrain_root_dir:str,
                                 pretrain_dir_name:str,
                                 pretrain_module_name:str
                                 ) -> List[str]:
        pretrain_dir = f"{pretrain_root_dir}/{pretrain_dir_name}"
        
        if pretrain_module_name in ["all","last_epoch"]:
            pretrain_name_list:List[str] = [
                pretrain_module
                for pretrain_module in os.listdir(pretrain_dir)
                if pretrain_module.endswith("pth") and "checkpoint" not in pretrain_module
                ]
            pretrain_name_list.sort()

            if pretrain_module_name == "last_epoch":
                pretrain_name_list = [pretrain_name_list[-1]]
        else:
            pretrain_name_list:List[str] = [pretrain_module_name]
        
        return [f"{pretrain_dir}/{pretrain_name}" for pretrain_name in pretrain_name_list]
    
    def pretrained_load(self,pretrain_path:str) -> None:
        if pretrain_path is None:
            return
        pretrained_load:dict = torch.load(pretrain_path,map_location='cpu')
        self.model.load_state_dict(pretrained_load)
        self.model = self.model.to(self.device)
        self.model.eval()