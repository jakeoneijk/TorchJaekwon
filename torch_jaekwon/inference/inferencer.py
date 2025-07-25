#type
from typing import List, Tuple,Union, Literal
from torch import Tensor
#package
import os
import torch
import torch.nn as nn
from tqdm import tqdm
#torchjaekwon
from torch_jaekwon import GetModule
from ..util import UtilData 
#internal

class Inferencer():
    def __init__(
        self,
        output_dir:str,
        save_dir_name:str,
        model:Union[nn.Module,object],
        model_class_meta:dict, #{name:[file_name, class_name], args: {}}
        set_type:Literal[ 'single', 'dir', 'testset' ],
        set_meta_dict: dict,
        device:torch.device,
    ) -> None:
        self.output_dir:str = output_dir
        self.save_dir_name:str = save_dir_name
        
        self.device:torch.device = device

        assert model_class_meta is not None or model is not None, "model_class_meta or model must be not None"
        self.model:Union[nn.Module,object] = self.get_model(model_class_meta) if model is None else model
        self.shared_dir_name:str = '_shared_'

        self.set_type:Literal[ 'single', 'dir', 'testset' ] = set_type
        self.set_meta_dict:dict = set_meta_dict
    
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

    def get_output_dir_path(self, pretrained_name:str, meta_data:dict) -> Tuple[str,str]:
        output_dir_path: str = f'''{self.output_dir}/{self.save_dir_name}({pretrained_name})/{meta_data["test_name"]}'''
        shared_output_dir_path:str = f'''{self.output_dir}/{self.shared_dir_name}/{meta_data["test_name"]}'''
        return output_dir_path, shared_output_dir_path
    
    def read_data_dict_by_meta_data(self, meta_data:dict) -> dict:
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
    
    def post_process(self, data_dict: dict) -> dict:
        return data_dict

    def save_data(self, output_dir_path:str, shared_output_dir_path:str, meta_data:dict, data_dict:dict) -> None:
        pass
    
    @torch.no_grad()
    def update_data_dict_by_model_inference(self, data_dict: dict) -> dict:
        if type(data_dict["model_input"]) == Tensor:
            data_dict["pred"] = self.model(data_dict["model_input"].to(self.device))
        return data_dict
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def get_model(self, model_class_meta:dict) -> nn.Module:
        model = GetModule.get_module(class_type = 'model', module_name = model_class_meta['name'], arg_dict=model_class_meta['args'])
        return model

    def inference(
        self,
        pretrained_root_dir:str,
        pretrained_dir_name:str,
        ckpt_name:str
    ) -> None:
        pretrained_path_list:List[str] = self.get_pretrained_path_list(
            pretrain_root_dir= pretrained_root_dir,
            pretrain_dir_name = pretrained_dir_name,
            ckpt_name= ckpt_name
        )

        for pretrained_path in pretrained_path_list:
            self.pretrained_load(pretrained_path) 
            pretrained_name:str = UtilData.get_file_name(file_path=pretrained_path)
            meta_data_list:List[dict] = self.get_inference_meta_data_list()
            for meta_data in tqdm(meta_data_list,desc='inference by meta data'):
                output_dir_path, shared_output_dir_path = self.get_output_dir_path(pretrained_name=pretrained_name,meta_data=meta_data)
                if output_dir_path is None: continue
                data_dict:dict = self.read_data_dict_by_meta_data(meta_data=meta_data)
                data_dict = self.update_data_dict_by_model_inference(data_dict)
                data_dict:dict = self.post_process(data_dict)

                self.save_data(output_dir_path, shared_output_dir_path, meta_data, data_dict)    
                    
    def get_pretrained_path_list(
        self,
        pretrain_root_dir:str,
        pretrain_dir_name:str,
        ckpt_name:str
    ) -> List[str]:
        pretrain_dir = f"{pretrain_root_dir}/{pretrain_dir_name}"
        
        if ckpt_name in ["all","last"]:
            pretrain_name_list:List[str] = [
                pretrain_module
                for pretrain_module in os.listdir(pretrain_dir)
                if os.path.splitext(pretrain_module)[-1] in [".pth"] and "checkpoint" not in pretrain_module
                ]
            pretrain_name_list.sort()

            if ckpt_name == "last":
                pretrain_name_list = [pretrain_name_list[-1]]
        else:
            pretrain_name_list:List[str] = [ckpt_name]
        
        return [f"{pretrain_dir}/{pretrain_name}" for pretrain_name in pretrain_name_list]
    
    def pretrained_load(self,pretrain_path:str) -> None:
        if pretrain_path is None:
            return
        pretrained_load:dict = torch.load(pretrain_path,map_location='cpu')
        key_list = list(pretrained_load.keys())
        for key in key_list: 
            if '_orig_mod.' in key: pretrained_load[key.replace('_orig_mod.', '')] = pretrained_load.pop(key)
        self.model.load_state_dict(pretrained_load)
        self.model.to(self.device)
        self.model.eval()