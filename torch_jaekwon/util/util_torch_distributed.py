from typing import Union
import os
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import torch.distributed as distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

def torchrun_setup() -> None:
    torch.cuda.set_device(local_rank())
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))

def is_available() -> bool:
    return distributed.is_initialized() and distributed.is_available()

def local_rank() -> int:
    local_rank:int = int(os.environ.get('LOCAL_RANK', 0))
    return local_rank

def world_size() -> int:
    world_size:int = int(os.environ.get('WORLD_SIZE', 1))
    return world_size

def is_main_process() -> bool:
    return local_rank() == 0

def barrier() -> None:
    distributed.barrier()

def model_to_ddp(model:Union[nn.Module, dict], gpu_id:int = 0, find_unused_parameters:bool = False) -> None:
    if isinstance(model, dict):
        for model_name in model:
            model[model_name] = model_to_ddp(model[model_name], gpu_id, find_unused_parameters=find_unused_parameters)
        return model
    else:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=find_unused_parameters)
        return model

def get_dataloader(dataloader_args:dict, shuffle:bool = True) -> DataLoader:
    dataset = dataloader_args['dataset']
    if not isinstance(dataset, IterableDataset):
        dataloader_args.pop("shuffle", None)
        dataloader_args['sampler'] = DistributedSampler(dataset, rank=local_rank(), shuffle=shuffle)
    return DataLoader(**dataloader_args)

def finish() -> None:
    distributed.barrier()
    distributed.destroy_process_group()