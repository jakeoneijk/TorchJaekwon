from typing import List

from TorchJaekwon.DataProcess.Preprocess.Preprocessor import Preprocessor

class TemplatePreprocessor(Preprocessor):
    def __init__(self, data_name:str) -> None:
        super().__init__(data_name)

    def get_meta_data_param(self) -> List[tuple]:
        '''
        meta_data_param_list = list()
        '''
        raise NotImplementedError

    def preprocess_one_data(self,param_dict: dict) -> None:
        '''
        ex) (subset, file_name) = param
        '''
        raise NotImplementedError