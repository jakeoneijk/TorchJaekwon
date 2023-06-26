import os
from torch.utils.data import DataLoader

from HParams import HParams
from TorchJaekwon.GetModule import GetModule

class PytorchDataLoader:
    def __init__(self):
        self.h_params = HParams()
        self.data_loader_config:dict = self.h_params.pytorch_data.dataloader
    
    def get_pytorch_data_loaders(self) -> dict:
        pytorch_dataset_dict = self.get_pytorch_data_set_dict()
        pytorch_data_loader_config_dict = self.get_pytorch_data_loader_config(pytorch_dataset_dict)
        pytorch_data_loader_dict = self.get_pytorch_data_loaders_from_config(pytorch_data_loader_config_dict)
        if getattr(self.h_params.data,'use_testset_as_validset',False):
            pytorch_data_loader_dict['valid'] = pytorch_data_loader_dict['test']
        return pytorch_data_loader_dict
    
    def get_pytorch_data_set_dict(self) -> dict:
        pytorch_dataset_dict = dict()
        for subset in self.data_loader_config:
            config_for_dataset = {
                "subset": subset
            }
            pytorch_dataset_dict[subset] = GetModule.get_module_class('./Data/PytorchDataset',self.data_loader_config[subset]["dataset"]["class_name"])(config_for_dataset)
        return pytorch_dataset_dict
    
    def get_pytorch_data_loader_config(self,pytorch_dataset:dict) -> dict:
        pytorch_data_loader_config_dict:dict = {subset:dict() for subset in pytorch_dataset}

        for subset in pytorch_dataset:
            args_exception_list = self.get_exception_list_of_dataloader_args_config(subset)
            pytorch_data_loader_config_dict[subset]["dataset"] = pytorch_dataset[subset]
            for arg_name in self.data_loader_config[subset]:
                if arg_name in args_exception_list:
                    continue
                if arg_name == 'batch_sampler':
                    arguments_for_args_class:dict = self.h_params.pytorch_data.dataloader[subset]['batch_sampler']
                    arguments_for_args_class.update({"pytorch_dataset":pytorch_dataset[subset],"subset":subset})
                    pytorch_data_loader_config_dict[subset][arg_name] = GetModule.get_module_class('./Data/PytorchDataLoader',
                                                                                             self.data_loader_config[subset][arg_name]["class_name"]
                                                                                             )(arguments_for_args_class)
                elif arg_name == 'collate_fn':
                    if self.data_loader_config[subset][arg_name] == True: pytorch_data_loader_config_dict[subset][arg_name] = pytorch_data_loader_config_dict[subset]["dataset"].collate_fn
                else:
                    pytorch_data_loader_config_dict[subset][arg_name] = self.data_loader_config[subset][arg_name]
        
        return pytorch_data_loader_config_dict
    
    def get_exception_list_of_dataloader_args_config(self,subset):
        args_exception_list = ["dataset"]
        if "batch_sampler" in self.data_loader_config[subset]:
            args_exception_list += ["batch_size", "shuffle", "sampler", "drop_last"]
        return args_exception_list

    def get_pytorch_data_loaders_from_config(self,dataloader_config:dict) -> dict:
        pytorch_data_loader_dict = dict()
        for subset in dataloader_config:
            pytorch_data_loader_dict[subset] = DataLoader(**dataloader_config[subset])
        return pytorch_data_loader_dict



    