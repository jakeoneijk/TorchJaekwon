from typing import Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset

class BalancedMultiDataset(IterableDataset):
    def __init__(
        self,
        sampling_schedule_dict:dict = None, # {'data1': 10, 'data2': 2}
        random_seed:int = (int)(torch.cuda.initial_seed() / (2**32)),
        is_random_seed_per_dataset:bool = True,
    ) -> None:
        self.data_list_dict: Dict[str,list] = self.init_data_list_dict() # {data_type1: List, data_type2: List}
        self.data_list_dict['data_name'] = list(self.data_list_dict.keys())
        
        self.idx_dict = {data_name: 0 for data_name in self.data_list_dict}
        self.idx_dict['data_name'] = -1 # it will start from 0 by adding 1

        self.sampling_schedule_dict = sampling_schedule_dict if sampling_schedule_dict is not None else {data_name: 1 for data_name in self.data_list_dict['data_name']}

        self.random_state_dict = dict()
        self.random_state_dict['data_name'] = np.random.RandomState(random_seed)
        self.random_state_dict['data_name'].shuffle(self.data_list_dict['data_name'])

        for data_name in self.data_list_dict['data_name']:
            self.random_state_dict[data_name] = np.random.RandomState(np.random.RandomState(random_seed).randint(low=0, high=10000) if is_random_seed_per_dataset else random_seed)
            self.random_state_dict[data_name].shuffle(self.data_list_dict[data_name])
            print("{}: {}".format(data_name, len(self.data_list_dict[data_name])))
    
    def init_data_list_dict(self) -> Dict[str,list]: # {data_type1: List, data_type2: List}
        raise NotImplementedError("You must implement the init_data_list_dict method in the subclass.")

    def read_data(self,meta_data):
        raise NotImplementedError("You must implement the read_data method in the subclass.")

    def __iter__(self):
        while True:
            self.increase_idx('data_name')
            data_name:str = self.data_list_dict['data_name'][self.idx_dict['data_name']]

            for _ in range(self.sampling_schedule_dict[data_name]):
                self.increase_idx(data_name)
                data = self.read_data(self.data_list_dict[data_name][self.idx_dict[data_name]])
                yield data   

    def increase_idx(self, key:str, max) -> None:
        self.idx_dict[key] += 1
        if self.idx_dict[key] == len(self.data_list_dict[key]):
            self.idx_dict[key] = 0
            self.random_state_dict[key].shuffle(self.data_list_dict[key])

    def __len__(self):
        return max([len(self.data_list_dict[data_name]) for data_name in self.data_list_dict])

    
