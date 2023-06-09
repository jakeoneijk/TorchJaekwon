from typing import Union,Dict,List
from numpy import ndarray
from torch import Tensor

import os
import copy
import numpy as np
import torch
import pickle
import yaml
import csv
from pathlib import Path

class UtilData:

    @staticmethod
    def get_file_name_from_path(path:str, with_ext:bool = False) -> str:
        if path is None:
            print("warning: path is None")
            return ""
        path_pathlib = Path(path)
        if with_ext:
            return path_pathlib.name
        else:
            return path_pathlib.stem
    
    @staticmethod
    def pickle_save(save_path:str, data:Union[ndarray,Tensor]) -> None:
        assert(os.path.splitext(save_path)[1] == ".pkl") , "file extension should be '.pkl'"

        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        
        with open(save_path,'wb') as file_writer:
            pickle.dump(data,file_writer)
    
    @staticmethod
    def pickle_load(data_path:str) -> Union[ndarray,Tensor]:
        with open(data_path, 'rb') as pickle_file:
            data:Union[ndarray,Tensor] = pickle.load(pickle_file)
        return data
    
    @staticmethod
    def yaml_save(save_path:str, data:Union[dict,list], sort_keys:bool = False) -> None:
        assert(os.path.splitext(save_path)[1] == ".yaml") , "file extension should be '.yaml'"

        with open(save_path, 'w') as file:
            yaml.dump(data, file, sort_keys = sort_keys)
    
    @staticmethod
    def yaml_load(data_path:str) -> dict:
        yaml_file = open(data_path, 'r')
        return yaml.safe_load(yaml_file)
    
    @staticmethod
    def csv_load(data_path:str) -> list:
        row_result_list = list()
        with open(data_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
            for row in spamreader:
                row_result_list.append(row)
        return row_result_list
    
    @staticmethod
    def csv_save(file_path:str,
                 data_dict_list:List[Dict[str,object]], #[ {key:object}, ... ]
                 order_of_key:list = None # [key1, key2, ...]
                 ) -> list:
        import pandas as pd
        if order_of_key is None:
            order_of_key = list(data_dict_list[0].keys())
        csv_save_dict:dict = {key:list() for key in order_of_key}
        for data_dict in data_dict_list:
            for key in csv_save_dict:
                csv_save_dict[key].append(data_dict[key])
        pd.DataFrame(csv_save_dict).to_csv(file_path)

    @staticmethod
    def save_data_segment(save_dir:str,data:ndarray,segment_len:int,segment_axis:int=-1,remainder:str = ['discard','pad','maintain'][1],ext:str = ['pkl'][0]):
        os.makedirs(save_dir,exist_ok=True)
        data_total = copy.deepcopy(data)
        total_length_of_data:int = data_total.shape[segment_axis]

        if total_length_of_data % segment_len != 0 and remainder in ['discard','pad']:
            if remainder == 'discard':
                data_total = data_total.take(indices=range(0, total_length_of_data - (total_length_of_data % segment_len)), axis=segment_axis)
            else:
                assert(segment_axis==-1 and (len(data_total.shape) in [1,2])),'Error[UtilData.save_data_segment] not implemented yet' 
                pad_length:int = segment_len - (total_length_of_data % segment_len)
                if len(data_total.shape) == 1:
                    data_total = np.pad(data_total, (0, pad_length), 'constant')
                elif len(data_total.shape) == 2:
                    data_total = np.pad(data_total, ((0,0),(0,pad_length)), 'constant')
            total_length_of_data:int = data_total.shape[segment_axis]
        
        for start_idx in range(0,total_length_of_data,segment_len):
            end_idx:int = start_idx + segment_len
            if remainder == 'maintain' and end_idx >= total_length_of_data: end_idx = total_length_of_data - 1
            
            data_segment = data_total.take(indices=range(start_idx, end_idx), axis=segment_axis)

            assert(data_segment.shape[segment_axis] == segment_len),'Error[UtilData.save_data_segment] segment length error!!'
            if ext == 'pkl':
                UtilData.pickle_save(f'{save_dir}/{start_idx}.{ext}',data_segment)
    
    @staticmethod
    def fit_feature_shape_length(feature:Union[Tensor,ndarray],shape_length:int) -> Tensor:
        if type(feature) != torch.Tensor:
            feature = torch.from_numpy(feature)

        for _ in range(shape_length - len(feature.shape)):
            feature = torch.unsqueeze(feature, 0)
        
        return feature
    
    @staticmethod
    def list_of_dict_to_dict_batch(input_dict:List[Dict[str,ndarray]],output_type:str = ['numpy','torch'][1]) -> Dict[str,ndarray]:
        batch_dict: Dict[str,list] = {feature_name: list() for feature_name in input_dict[0]}
        for feature_dict in input_dict:
            for feature_name in feature_dict:
                batch_dict[feature_name].append(feature_dict[feature_name])
        for feature_name in batch_dict:
            if output_type == 'numpy':
                batch_dict[feature_name] = np.array(batch_dict[feature_name])
            elif output_type == 'torch':
                batch_dict[feature_name] = torch.from_numpy(np.array(batch_dict[feature_name]))
        return batch_dict
    
    @staticmethod
    def sort_dict_list_by_key(dict_list:List[dict], key:str):
        return sorted(dict_list, key=lambda dictionary: dictionary[key])
        
