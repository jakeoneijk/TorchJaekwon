from typing import List

from TorchJAEKWON.DataProcess.Preprocess.Preprocessor import Preprocessor

class TemplatePreprocessor(Preprocessor):
    def __init__(self, data_config_dict: dict = None) -> None:
        super().__init__(data_config_dict)
    
    def get_dataset_name(self) -> str:
        raise NotImplementedError

    def get_meta_data_param(self) -> List[tuple]:
        '''
        meta_data_param_list = list()
        '''
        raise NotImplementedError

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