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

from . import util

def log(text:str, main_only:bool = True, **kwargs) -> None:
    if is_available():
        kwargs['prefix'] = f"{local_rank()}/{world_size() - 1}: " + kwargs.get('prefix', '')
    if not main_only or is_main_process():
        util.log(text, **kwargs)

def torchrun_setup() -> None:
    torch.cuda.set_device(f'cuda:{local_rank()}')
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

def model_to_ddp(model:Union[nn.Module, dict], gpu_id:int = 0, find_unused_parameters:bool = True) -> None:
    if isinstance(model, dict):
        for model_name in model:
            model[model_name] = model_to_ddp(model[model_name], gpu_id, find_unused_parameters=find_unused_parameters)
        return model
    else:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=find_unused_parameters)
        return model

def get_dataloader(dataloader_args:dict, shuffle:bool = True) -> DataLoader:
    effective_batch_size:int = dataloader_args.get('batch_size', 1)
    assert effective_batch_size % world_size() == 0, f"Batch size {effective_batch_size} must be divisible by world size {world_size()}."
    dataloader_args['batch_size'] = effective_batch_size // world_size()
    log(f'Effective batch size: {effective_batch_size} | Per-process batch size: {dataloader_args["batch_size"]}', msg_type='info')
    
    dataset = dataloader_args['dataset']
    if not isinstance(dataset, IterableDataset):
        dataloader_args.pop("shuffle", None)
        dataloader_args['sampler'] = DistributedSampler(dataset, rank=local_rank(), shuffle=shuffle)
    return DataLoader(**dataloader_args)

def shard_list(data: list, shard_index: int = None, num_shards: int = None) -> list:
    if shard_index is None:
        shard_index = local_rank()
    if num_shards is None:
        num_shards = world_size()
    return data[shard_index::num_shards]

def finish() -> None:
    distributed.barrier()
    distributed.destroy_process_group()