#type
from typing import List,Dict,Union
#package
import os
from tqdm import tqdm
import numpy as np
import copy

from HParams import HParams
from TorchJaekwon.Util.UtilData import UtilData

class Evaluater():
    def __init__(self,
                 source_dir:str
                 ) -> None:
        self.source_dir:str = source_dir
        self.evaluation_result_dir:str = f"{self.source_dir}/evaluation"
        os.makedirs(self.evaluation_result_dir,exist_ok=True)
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    def get_meta_data_list(self) -> Union[List[dict],Dict[str,list]]:
        pass

    def get_result_dict_from_one_meta_data(
        self,
        meta_data:dict
        ) -> dict: #{'name':name_of_testcase,'metric_name1':value1,'metric_name2':value2... }
        pass
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    def evaluate(self) -> None:
        meta_data_list:Union[List[dict],Dict[str,list]] = self.get_meta_data_list()
        self.get_result_dict(meta_data_list)
    
    def get_result_dict(self,meta_data_list:Union[List[dict],Dict[str,list]]) -> dict:
        return self.get_result_dict_from_meta_data_list(meta_data_list)
    
    def get_result_dict_from_meta_data_list(self,
                                            meta_data_list:list):
        result_dict_list:List[dict] = list()
        for meta_data in tqdm(meta_data_list,desc='get result'):
            result_dict_list.append(self.get_result_dict_from_one_meta_data(meta_data))
        
        metric_name_list:list = [metric_name for metric_name in list(result_dict_list[0].keys()) if type(result_dict_list[0][metric_name]) in [float,np.float_]]
        metric_name_list.sort()
        mean_median_std_dict:dict = self.get_mean_median_std_from_dict_list(result_dict_list,metric_name_list)
        
        UtilData.yaml_save(f'{self.evaluation_result_dir}/mean_median_std.yaml',mean_median_std_dict)

        for metric_name in metric_name_list:
            UtilData.yaml_save(f'{self.evaluation_result_dir}/sort_by_{metric_name}.yaml',UtilData.sort_dict_list_by_key(result_dict_list,metric_name))
    
    def get_mean_median_std_from_dict_list(self,dict_list:List[dict],metric_name_list:List[str]):
        result_list_dict:dict = {metric_name: list() for metric_name in metric_name_list}
        for result in dict_list:
            for metric_name in metric_name_list:
                result_list_dict[metric_name].append(result[metric_name])
        result_dict = dict()
        for metric_name in metric_name_list:
            result_dict[metric_name] = dict()
            result_dict[metric_name]['mean'] = float(np.mean(result_list_dict[metric_name]))
            result_dict[metric_name]['median'] = float(np.median(result_list_dict[metric_name]))
            result_dict[metric_name]['std'] = float(np.std(result_list_dict[metric_name]))
        return result_dict