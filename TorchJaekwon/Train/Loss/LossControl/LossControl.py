from TorchJaekwon.GetModule import GetModule
from HParams import HParams
import torch

class LossControl:
    def __init__(self) -> None:
        self.h_params = HParams()
        self.loss_meta_dict : dict = self.h_params.train.loss_dict
        self.loss_function_dict : dict = dict()
        self.set_loss_function_dict()
    
    def set_loss_function_dict(self) -> None:
        '''
        self.loss_meta_dict:
                            name: 
                                pred:
                                target:
                                lossfunction:
                                weight:
        '''

        for loss_name in self.loss_meta_dict:
            lossfunction:str = self.loss_meta_dict[loss_name]["lossfunction"]
            if lossfunction == "mse":
                self.loss_function_dict[loss_name] = torch.nn.MSELoss()
            elif lossfunction == "l1":
                self.loss_function_dict[loss_name] = torch.nn.L1Loss()
            elif lossfunction == "l2":
                self.loss_function_dict[loss_name] = torch.nn.MSELoss()
            elif lossfunction == "bce":
                self.loss_function_dict[loss_name] = torch.nn.BCELoss()
            elif lossfunction == "cross_entropy":
                self.loss_function_dict[loss_name] = torch.nn.CrossEntropyLoss()
            else:
                get_module = GetModule()
                loss_args:dict = self.loss_meta_dict[loss_name]["loss_args"]
                self.loss_function_dict[loss_name] = get_module.get_module(
                    module_type="loss_function",
                    module_name=lossfunction,
                    module_arg=loss_args,arg_unpack=True)
    
    def set_loss_function_dict_additional(self,loss_name:str, lossfunction_name:str) -> None:
        pass

    def to(self,device) -> None:
        for loss_name in self.loss_function_dict:
            self.loss_function_dict[loss_name] = self.loss_function_dict[loss_name].to(device)
    
    def cuda(self,device) -> None:
        for loss_name in self.loss_function_dict:
            self.loss_function_dict[loss_name].cuda(device)
    
    def get_loss_function_name_list(self) -> list:
        return list(self.loss_function_dict.keys()) + ["total_loss"]
    
    def calculate_total_loss_by_loss_meta_dict(self, pred_dict:dict, target_dict:dict, loss_name_list:list = [], final_loss_name:str = "total_loss") -> dict:
        loss_dict:dict = dict()

        if loss_name_list == []:
            loss_name_list = list(self.loss_meta_dict.keys())

        for loss_name in loss_name_list:
            pred_name:str = self.loss_meta_dict[loss_name]['pred']
            target_name:str = self.loss_meta_dict[loss_name]['target']
            weight = self.loss_meta_dict[loss_name]['weight']

            if type(pred_dict) is list:
                pred_dict_unpack = list()
                for pred in pred_dict:
                    pred_dict_unpack.append(pred[pred_name])
                loss_dict[loss_name] = self.loss_function_dict[loss_name](pred_dict_unpack,target_dict[target_name]) * weight

            else:
                loss_dict[loss_name] = self.loss_function_dict[loss_name](pred_dict[pred_name],target_dict[target_name]) * weight

            if final_loss_name in loss_dict:
                loss_dict[final_loss_name] = loss_dict[final_loss_name] + loss_dict[loss_name]
            else:
                loss_dict[final_loss_name] = loss_dict[loss_name]
        return loss_dict
    