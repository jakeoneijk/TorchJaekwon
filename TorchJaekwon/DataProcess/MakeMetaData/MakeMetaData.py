import os
from abc import ABC, abstractmethod
from HParams import HParams

class MakeMetaData(ABC):

    def __init__(self,make_meta_data_config:dict) -> None:
        self.h_params: HParams = HParams()
        self.data_name_list = list()
        for data_name in self.h_params.data.data_config_per_dataset_dict:
            if self.h_params.data.data_config_per_dataset_dict[data_name]["load_to_pytorch_dataset"]:
                self.data_name_list.append(data_name)
        
        self.data_root_path_list = list()
        for data_name in self.data_name_list:
            self.data_root_path_list.append(os.path.join(self.h_params.data.root_path,data_name))

        self.make_meta_data_config:dict = make_meta_data_config
    
    @abstractmethod
    def make_meta_data(self):
        raise NotImplementedError