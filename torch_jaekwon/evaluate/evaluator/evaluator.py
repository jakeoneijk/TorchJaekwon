#type
from typing import List
#import
import os
from tqdm import tqdm
import numpy as np
import torch
#torchjaekwon import
from ...util import util_data, util_torch
#internal import

class Evaluator():
    def __init__(
        self,
        pred_dir_path:str,
        gt_dir_path:str,
        evaluation_result_dir:str,
        batch_size:int = 1, 
        sort_result_by_metric:bool = True,
        device:torch.device = torch.device('cpu')
    ) -> None:
        self.pred_dir_path:str = pred_dir_path
        self.gt_dir_path:str = gt_dir_path
        self.evaluation_result_dir:str = f'{evaluation_result_dir}/{util_data.get_file_name(self.pred_dir_path)}'
        self.batch_size:int = batch_size
        self.sort_result_by_metric = sort_result_by_metric
        self.device:torch.device = device
        os.makedirs(self.evaluation_result_dir,exist_ok=True)
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    def get_eval_dir_list(self) -> List[str]:
        return [self.pred_dir_path]

    def get_meta_data_list(self, eval_dir:str) -> List[dict]:
        pass

    def get_sample_wise_result(
        self,
        meta_data:dict
    ) -> dict: #{'name':name_of_testcase,'metric_name1':value1,'metric_name2':value2... }
        pass

    def get_set_wise_result(self, meta_data_list:List[dict]) -> dict:
        return {'result':dict()}
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    def evaluate(self) -> None:
        eval_dir_list:List[str] = self.get_eval_dir_list()

        for eval_dir in tqdm(eval_dir_list, desc='evaluate eval dir'):
            meta_data_list: List[dict] = self.get_meta_data_list(eval_dir)
            result_dict:dict = self.get_result_dict(meta_data_list)

            test_set_name:str = eval_dir.split('/')[-1]
            util_data.yaml_save(f'{self.evaluation_result_dir}/{test_set_name}.yaml',result_dict['result'])
            if self.sort_result_by_metric:
                for metric_name in result_dict['result']:
                    util_data.yaml_save(f'{self.evaluation_result_dir}/{test_set_name}_sort_by_{metric_name}.yaml',util_data.sort_dict_list( dict_list = result_dict['result_per_sample'], key = metric_name))
    
    def get_result_dict(self,meta_data_list:List[dict]) -> dict:
        result_dict:dict = self.get_set_wise_result(meta_data_list)

        result_dict['result_per_sample'] = list()
        if self.batch_size > 1:
            meta_data_list = util_torch.chunk_list(meta_data_list, self.batch_size)
        for meta_data in tqdm(meta_data_list,desc='get result'):
            result = self.get_sample_wise_result(meta_data)
            if not isinstance(result, list): result = [result]
            result_dict['result_per_sample'] += result

        metric_name_list:list = [metric_name for metric_name in list(result_dict['result_per_sample'][0].keys()) if type(result_dict['result_per_sample'][0][metric_name]) in [float]]
        metric_name_list.sort()
        mean_median_std_dict:dict = self.get_mean_median_std_from_dict_list(result_dict['result_per_sample'], metric_name_list)

        result_dict['result'].update(mean_median_std_dict)
        return result_dict
    
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