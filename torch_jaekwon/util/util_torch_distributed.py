from typing import Dict
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def get_config() -> Dict[str, int]:
    local_rank:int = int(os.environ.get('LOCAL_RANK', 0))
    world_size:int = int(os.environ.get('WORLD_SIZE', 1))
    return {'local_rank': local_rank, 'world_size': world_size}

def get_dataloader(dataloader_args:dict, local_rank:int, shuffle:bool = True) -> DataLoader:
    dataset = dataloader_args['dataset']
    dataloader_args['sampler'] = DistributedSampler(dataset, rank=local_rank, shuffle=shuffle)
    return DataLoader(**dataloader_args)