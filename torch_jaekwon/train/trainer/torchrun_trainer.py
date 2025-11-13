from typing import Literal, Union
import os
import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ...util import util_torch_distributed, util_torch
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
        super().__init__(device = util_torch_distributed.local_rank(), *args, **kwargs)

        self.model = util_torch_distributed.model_to_ddp(self.model, gpu_id=self.device)
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
    
    def log_current_state(self,train_state:TrainState = None, is_log_media:bool = True) -> None:
        if not util_torch_distributed.is_main_process(): return
        return super().log_current_state(train_state, is_log_media)

    def save_checkpoint(self, save_name:str = 'train_checkpoint.pth') -> str:
        if not util_torch_distributed.is_main_process(): return None
        return super().save_checkpoint(save_name)
    
    def save_module(self, model, model_name = '', name = 'pretrained_best_epoch') -> None:
        if not util_torch_distributed.is_main_process():
            return
        if isinstance(model, dict):
            for model_type in model:
                self.save_module(model[model_type], model_name + f'{model_type}_', name)
        else:
            path = os.path.join(self.logger.log_path["root"],f'{model_name}{name}.pth')
            if self.model_ema is not None:
                raise NotImplementedError("EMA model saving is not implemented in distributed training.")
            else:
                state_dict = model.module.state_dict()
            torch.save(state_dict, path)
    
    def get_state_dict(self, module:Union[dict, nn.Module]) -> Union[dict, nn.Module]:
        if isinstance(module, DDP):
            return module.module.state_dict()
        elif hasattr(module, 'state_dict'):
            return module.state_dict()
        elif isinstance(module, dict):
            state_dict = dict()
            for key in module:
                state_dict[key] = self.get_state_dict(module[key])
            return state_dict
        else:
            raise ValueError(f'Cannot get state_dict from {module}')
    
    def fit(self) -> None:
        super().fit()
        util_torch_distributed.finish()