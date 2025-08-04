from typing import Dict
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def local_rank() -> int:
    local_rank:int = int(os.environ.get('LOCAL_RANK', 0))
    return local_rank

def world_size() -> int:
    world_size:int = int(os.environ.get('WORLD_SIZE', 1))
    return world_size

def is_main_process() -> bool:
    return local_rank() == 0

def get_dataloader(dataloader_args:dict, shuffle:bool = True) -> DataLoader:
    dataset = dataloader_args['dataset']
    dataloader_args['sampler'] = DistributedSampler(dataset, rank=local_rank(), shuffle=shuffle)
    return DataLoader(**dataloader_args)