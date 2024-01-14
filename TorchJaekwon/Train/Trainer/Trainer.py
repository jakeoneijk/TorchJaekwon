from typing import Dict, Union, Literal

import os
from abc import ABC, abstractmethod
from enum import Enum,unique
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from HParams import HParams
from TorchJaekwon.GetModule import GetModule
from TorchJaekwon.Data.PytorchDataLoader.PytorchDataLoader import PytorchDataLoader
from TorchJaekwon.Train.LogWriter.LogWriter import LogWriter
from TorchJaekwon.Train.Optimizer.OptimizerControl import OptimizerControl
from TorchJaekwon.Train.AverageMeter import AverageMeter
from TorchJaekwon.Train.Loss.LossControl.LossControl import LossControl


@unique
class TrainState(Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"
 
class Trainer(ABC):
    def __init__(self,
                 #resource
                 device:torch.device = HParams().resource.device,
                 #class_meta
                 model_class_name:Union[str, list] = HParams().model.class_name,
                 model_class_meta_dict:dict = HParams().model.class_meta_dict,
                 loss_class_meta:dict = HParams().train.loss_control['class_meta'],
                 #train params
                 max_norm_value_for_gradient_clip:float = getattr(HParams().train,'max_norm_value_for_gradient_clip',None),
                 #train setting
                 save_model_every_step:int = getattr(HParams().train, 'save_model_every_step', None),
                 ) -> None:
        self.h_params = HParams()
        self.device:torch.device = device

        self.model_class_name:Union[str, list] = model_class_name
        self.model_class_meta_dict:dict = model_class_meta_dict
        self.model:Union[nn.Module, list, dict] = None
        
        self.optimizer_control:OptimizerControl = None
        self.loss_control: LossControl = None
        self.loss_class_meta:dict = loss_class_meta

        self.data_loader_dict:dict() = {subset: None for subset in ['train','valid','test']}

        self.seed:int = (int)(torch.cuda.initial_seed() / (2**32)) if self.h_params.train.seed is None else self.h_params.train.seed
        self.set_seeds(self.h_params.train.seed_strict)

        self.max_norm_value_for_gradient_clip:float = max_norm_value_for_gradient_clip

        self.current_epoch:int = 1
        self.total_epoch:int = self.h_params.train.epoch
        self.global_step:int = 0
        self.local_step:int = 0
        self.best_valid_metric:dict[str,AverageMeter] = None
        self.best_valid_epoch:int = 0
        self.save_model_every_step:int = save_model_every_step

        
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    
    @abstractmethod
    def run_step(self,data,metric,train_state:TrainState):
        """
        run 1 step
        1. get data
        2. use model

        3. calculate loss
            current_loss_dict = self.loss_control.calculate_total_loss_by_loss_meta_dict(pred_dict=pred, target_dict=train_data_dict)
        
        4. put the loss in metric (append)
            for loss_name in current_loss_dict:
                metric[loss_name].update(current_loss_dict[loss_name].item(),batch_size)

        return current_loss_dict["total_loss"],metric
        """
        raise NotImplementedError

    def save_best_model(self,prev_best_metric, current_metric):
        """
        compare what is the best metric
        If current_metric is better, 
            1.save best model
            2. self.best_valid_epoch = self.current_epoch
        Return
            better metric
        """
        if prev_best_metric is None:
            return current_metric
        
        if prev_best_metric[self.loss_control.final_loss_name].avg > current_metric[self.loss_control.final_loss_name].avg:
            self.save_module()
            self.best_valid_epoch = self.current_epoch
            return current_metric
        else:
            return prev_best_metric
    
    def log_metric(
        self, 
        metrics:Dict[str,AverageMeter],
        data_size: int,
        train_state=TrainState.TRAIN
        )->None:
        """
        log and visualizer log
        """
        if train_state == TrainState.TRAIN:
            x_axis_name:str = "step_global"
            x_axis_value:int = self.global_step
        else:
            x_axis_name:str = "epoch"
            x_axis_value:int = self.current_epoch

        log:str = f'Epoch ({train_state.value}): {self.current_epoch:03} ({self.local_step}/{data_size}) global_step: {self.global_step}\t'
        
        for metric_name in metrics:
            val:float = metrics[metric_name].avg
            log += f' {metric_name}: {val:.06f}'
            self.log_writer.visualizer_log(
                x_axis_name=x_axis_name,
                x_axis_value=x_axis_value,
                y_axis_name=f'{train_state.value}/{metric_name}',
                y_axis_value=val
            )
        self.log_writer.print_and_log(log)
    
    @torch.no_grad()
    def log_media(self) -> None:
        pass

    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    def set_seeds(self,strict=False):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            if strict:
                torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        random.seed(self.seed)

    def init_train(self, dataset_dict=None):
        self.init_model()
        self.init_optimizer()
        self.init_loss()
        self.model_to_device()
        
        self.log_writer:LogWriter = LogWriter(model=self.model)
        self.set_data_loader(dataset_dict)
    
    def init_model(self) -> None:
        if isinstance(self.model_class_name, list):
            self.model = dict()
            for class_name in self.model_class_name:
                self.model[class_name] = GetModule.get_model(class_name)
        else:   
            self.model:nn.Module = GetModule.get_model(self.h_params.model.class_name)
    
    def init_optimizer(self) -> None:
        self.optimizer_control = GetModule.get_module_class('./Train/Optimizer',self.h_params.train.optimizer_control_config['class_name'])(**{"model":self.model})
    
    def init_loss(self) -> None:
        self.loss_control = GetModule.get_module_class("./Train/Loss/LossControl", self.loss_class_meta['name'])()
    
    def model_to_device(self):
        if self.h_params.resource.multi_gpu:
            from TorchJaekwon.Train.Trainer.Parallel import DataParallelModel, DataParallelCriterion
            self.model = DataParallelModel(self.model)
            self.model.cuda()
            for loss_name in self.loss_control.loss_function_dict:
                self.loss_control.loss_function_dict[loss_name] = DataParallelCriterion(self.loss_control.loss_function_dict[loss_name])
        else:
            self.loss_control.to(self.h_params.resource.device)
            if isinstance(self.model_class_name, list):
                for class_name in self.model_class_name: 
                    self.model[class_name] = self.model[class_name].to(self.device)
            else:   
                self.model = self.model.to(self.device)

    def data_dict_to_device(self,data_dict:dict) -> dict:
        for feature_name in data_dict:
            data_dict[feature_name] = data_dict[feature_name].float().to(self.device)
        return data_dict
    
    def set_data_loader(self,dataset_dict=None):
        data_loader_getter:PytorchDataLoader = GetModule.get_module_class('./Data/PytorchDataLoader', self.h_params.pytorch_data.class_meta['name'])()

        if dataset_dict is not None:
            pytorch_data_loader_config_dict = data_loader_getter.get_pytorch_data_loader_config(dataset_dict)
            self.data_loader_dict = data_loader_getter.get_pytorch_data_loaders_from_config(pytorch_data_loader_config_dict)
        else:
            self.data_loader_dict = data_loader_getter.get_pytorch_data_loaders()
    
    def fit(self):
        if getattr(self.h_params.train,'check_evalstep_first',False):
            print("check evaluation step first whether there is no error")
            with torch.no_grad():
                valid_metric = self.run_epoch(self.data_loader_dict['valid'],TrainState.VALIDATE, metric_range = "epoch")
                
        for _ in range(self.current_epoch, self.total_epoch):
            self.log_writer.print_and_log(f'----------------------- Start epoch : {self.current_epoch} / {self.h_params.train.epoch} -----------------------')
            self.log_writer.print_and_log(f'current best epoch: {self.best_valid_epoch}')
            if self.best_valid_metric is not None:
                for loss_name in self.best_valid_metric:
                    self.log_writer.print_and_log(f'{loss_name}: {self.best_valid_metric[loss_name].avg}')
            
            self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------')
            self.log_writer.print_and_log(f'current lr: {self.get_current_lr()}')
            self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------')
    
            #Train
            self.log_writer.print_and_log('train_start')
            self.run_epoch(self.data_loader_dict['train'],TrainState.TRAIN, metric_range = "step")
            
            #Valid
            self.log_writer.print_and_log('valid_start')

            with torch.no_grad():
                valid_metric = self.run_epoch(self.data_loader_dict['valid'],TrainState.VALIDATE, metric_range = "epoch")
                self.lr_scheduler_step(call_state='epoch', args=valid_metric)
            
            self.best_valid_metric = self.save_best_model(self.best_valid_metric, valid_metric)

            if self.current_epoch > self.h_params.train.save_model_after_epoch and self.current_epoch % self.h_params.train.save_model_every_epoch == 0:
                self.save_module(name=f"pretrained_epoch{str(self.current_epoch).zfill(8)}")
            
            self.current_epoch += 1
            self.log_writer.log_every_epoch(model=self.model)

        self.log_writer.print_and_log(f'best_epoch: {self.best_valid_epoch}')
        self.log_writer.print_and_log('Training complete')
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState, metric_range:str = "step"):
        assert metric_range in ["step","epoch"], "metric range should be 'step' or 'epoch'"

        if train_state == TrainState.TRAIN:
            self.set_model_train_valid_mode('train')
        else:
            self.set_model_train_valid_mode('valid')

        dataset_size = len(dataloader)

        if metric_range == "epoch":
            metric = self.metric_init()

        for step,data in enumerate(dataloader):

            if metric_range == "step":
                metric = self.metric_init()

            if step >= len(dataloader):
                break

            self.local_step = step
            loss,metric = self.run_step(data,metric,train_state)
        
            if train_state == TrainState.TRAIN:
                self.backprop(loss)
                
                if self.local_step % self.h_params.log.log_every_local_step == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                
                self.global_step += 1

                self.lr_scheduler_step(call_state='step')
            
            if self.save_model_every_step is not None and self.global_step % self.save_model_every_step == 0:
                self.save_module(name=f"step{self.global_step}")
                self.log_current_state()
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)

        self.log_current_state(train_state)

        return metric
    
    def log_current_state(self,train_state:TrainState = None) -> None:
        if train_state == TrainState.TRAIN or train_state == None:
            self.save_checkpoint()
            self.save_checkpoint("train_checkpoint_backup.pth")
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST or train_state == None:
            with torch.no_grad():
                self.log_media()
    
    def backprop(self,loss):
        if self.max_norm_value_for_gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm_value_for_gradient_clip)

        if getattr(self.h_params.train,'optimizer_step_unit',1) == 1:
            self.optimizer_control.optimizer_zero_grad()
            loss.backward()
            self.optimizer_control.optimizer_step()
        else:
            loss.backward()
            if (self.global_step + 1) % self.h_params.train.optimizer_step_unit == 0:
                self.optimizer_control.step()
                self.optimizer_control.zero_grad()
    
    def set_model_train_valid_mode(self, mode: Literal['train','valid']):
        if mode == 'train':
            if isinstance(self.model, dict):
                for model_name in self.model:
                    self.model[model_name].train()
            elif isinstance(self.model, list):
                for i in range(len(self.model)):
                    self.model[i].train()
            else:
                self.model.train()
        else:
            if isinstance(self.model, dict):
                for model_name in self.model:
                    self.model[model_name].eval()
                    self.model[model_name].zero_grad()
            elif isinstance(self.model, list):
                for i in range(len(self.model)):
                    self.model[i].eval()
                    self.model[i].zero_grad()
            else:
                self.model.eval()
                self.model.zero_grad
    
    def metric_init(self):
        loss_name_list = self.loss_control.get_loss_function_name_list()
        initialized_metric = dict()

        for loss_name in loss_name_list:
            initialized_metric[loss_name] = AverageMeter()

        return initialized_metric

    def save_module(self,name = 'pretrained_best_epoch'):
        if isinstance(self.model, dict):
            for model_name in self.model:
                path = os.path.join(self.log_writer.log_path["root"],f'{model_name}_{name}.pth')
                torch.save(self.model[model_name].state_dict() if not self.h_params.resource.multi_gpu else self.model[model_name].module.state_dict(), path)
        elif isinstance(self.model, list):
            for i in range(len(self.model)):
                path = os.path.join(self.log_writer.log_path["root"],f'model_{i}_{name}.pth')
                torch.save(self.model[i].state_dict() if not self.h_params.resource.multi_gpu else self.model[i].module.state_dict(), path)
        else:
            path = os.path.join(self.log_writer.log_path["root"],f'{name}.pth')
            torch.save(self.model.state_dict() if not self.h_params.resource.multi_gpu else self.model.module.state_dict(), path)

    def load_module(self,name = 'pretrained_best_epoch'):
        path = os.path.join(self.log_writer.log_path["root"],f'{name}.pth')
        best_model_load = torch.load(path)
        self.model.load_state_dict(best_model_load)
    
    def get_current_lr(self):
        return self.optimizer_control.get_lr()
    
    def lr_scheduler_step(self, call_state:Literal['step','epoch'], args = None):
        if args is not None:
            self.optimizer_control.lr_scheduler_step(interval_type=call_state,args=args)
        else:
            self.optimizer_control.lr_scheduler_step(interval_type=call_state)
    
    def save_checkpoint(self,save_name:str = 'train_checkpoint.pth'):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'optimizers': self.optimizer_control.optimizer_state_dict(),
            'lr_scheduler': self.optimizer_control.lr_scheduler_state_dict(),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }

        model_dict = dict()
        if isinstance(self.model, dict):
            for model_name in self.model:
                model_dict[f'model_{model_name}'] = self.model[model_name].state_dict() if not self.h_params.resource.multi_gpu else self.model[model_name].module.state_dict()
        elif isinstance(self.model, list):
            for i in range(len(self.model)):
                model_dict[f'model_{i}'] = self.model[i].state_dict() if not self.h_params.resource.multi_gpu else self.model[i].module.state_dict()
        else:
            model_dict['model'] = self.model.state_dict() if not self.h_params.resource.multi_gpu else self.model.module.state_dict()
        
        train_state.update(model_dict)

        path = os.path.join(self.log_writer.log_path["root"],save_name)
        self.log_writer.print_and_log(save_name)
        torch.save(train_state,path)

    def load_train(self,filename:str):
        cpt = torch.load(filename,map_location='cpu')
        self.seed = cpt['seed']
        self.set_seeds(self.h_params.train.seed_strict)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']

        self.model = self.model.to(torch.device('cpu'))
        self.model.load_state_dict(cpt['models'])
        self.model = self.model.to(self.h_params.resource.device)

        self.optimizer_control.optimizer_load_state_dict(cpt['optimizers'])
        self.optimizer_control.lr_scheduler_load_state_dict(cpt['lr_scheduler'])
        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']