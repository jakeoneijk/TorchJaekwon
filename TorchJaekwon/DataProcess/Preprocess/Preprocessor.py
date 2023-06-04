from typing import List

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import os
import time
from tqdm import tqdm

from HParams import HParams

class Preprocessor(ABC):
    def __init__(self, data_config_dict:dict = None) -> None:
        self.h_params: HParams = HParams()
        self.data_name:str = self.get_dataset_name()
        self.preprocessed_data_path = os.path.join(self.h_params.data.root_path,self.data_name)
        self.data_config_dict:dict = data_config_dict
    
    def write_message(self,message_type:str,message:str) -> None:
        with open(f"{self.preprocessed_data_path}/{message_type}.txt",'a') as file_writer:
            file_writer.write(message+'\n')
    
    def preprocess_data(self) -> None:
        meta_param_list:list = self.get_meta_data_param()
        start_time:float = time.time()

        if self.h_params.preprocess.multi_processing:
            with ProcessPoolExecutor(max_workers=self.h_params.preprocess.max_workers) as pool:
                pool.map(self.preprocess_one_data, meta_param_list)
        else:
            for meta_param in tqdm(meta_param_list,desc='preprocess data'):
                self.preprocess_one_data(meta_param)

        self.finish_pre_process()
        print("{:.3f} s".format(time.time() - start_time))

    @abstractmethod
    def get_dataset_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_meta_data_param(self) -> List[tuple]:
        '''
        meta_data_param_list = list()
        '''
        raise NotImplementedError
    
    @abstractmethod
    def preprocess_one_data(self,param: tuple) -> None:
        '''
        ex) (subset, file_name) = param
        '''
        raise NotImplementedError

    def extract_features(self,input_feature:str) -> dict:
        '''
        extract features from input feature
        '''
        raise NotImplementedError
    
    def finish_pre_process(self) -> None:
        print("Finish preprocess")