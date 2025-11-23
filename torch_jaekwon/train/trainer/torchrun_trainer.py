from typing import Literal, Union
import os
import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ...util import util_torch_distributed
from .trainer import Trainer, TrainState

class TorchrunTrainer(Trainer):
    def __init__(
        self, 
        *args, 
        **kwargs
    ) -> None:
        assert not kwargs.get("use_ema", False), "EMA is not supported in TorchrunTrainer yet."
        util_torch_distributed.torchrun_setup()
        optimizer_class_meta_dict = kwargs.pop('optimizer_class_meta_dict')
        lr_scheduler_class_meta_dict = kwargs.pop('lr_scheduler_class_meta_dict', None)
        super().__init__(device = f'cuda:{util_torch_distributed.local_rank()}', *args, **kwargs)

        self.model = util_torch_distributed.model_to_ddp(self.model, gpu_id=util_torch_distributed.local_rank())
        self.optimizer = self.init_optimizer(optimizer_class_meta_dict)
        self.lr_scheduler = self.init_lr_scheduler(self.optimizer, lr_scheduler_class_meta_dict)
    
    def get_data_loader(self, dataset:dataset.Dataset, data_loader_args:dict, subset_name:Literal['train','valid','test']) -> DataLoader:
        if subset_name == 'train':
            return util_torch_distributed.get_dataloader(dataloader_args = {'dataset': dataset, **data_loader_args}, shuffle=data_loader_args.get("shuffle", True))
        else:
            return DataLoader(dataset=dataset, **data_loader_args)
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState, metric_range:str = "step") -> dict:
        if train_state == TrainState.TRAIN and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(self.current_epoch)
        return super().run_epoch(dataloader, train_state, metric_range)
    
    def loss_backward(self, loss: torch.Tensor) -> torch.Tensor:
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            loss.backward()
            '''
            import torch.distributed as dist
            loss_for_log = loss.detach()
            dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
            return loss_for_log
            '''
        else:
            with self.model.no_sync():
                loss.backward()
        return loss
    
    def load_train(self, filename:str, map_location:str = 'cpu') -> None:
        map_location = 'cuda:%d' % util_torch_distributed.local_rank()
        super().load_train(filename, map_location={'cuda:0': map_location})
    
    def fit(self) -> None:
        super().fit()
        util_torch_distributed.finish()