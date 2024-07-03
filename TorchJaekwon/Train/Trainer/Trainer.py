#type
from typing import Dict, Union, Literal, Type
from enum import Enum,unique
#import
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#torchjaekwon import
from TorchJaekwon.GetModule import GetModule
from TorchJaekwon.Data.PytorchDataLoader.PytorchDataLoader import PytorchDataLoader
from TorchJaekwon.Train.LogWriter.LogWriter import LogWriter
from TorchJaekwon.Util.UtilData import UtilData
from TorchJaekwon.Train.AverageMeter import AverageMeter
#internal import
from HParams import HParams

@unique
class TrainState(Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"
 
class Trainer():
    def __init__(self,
                 #resource
                 device:torch.device,
                 #class_meta
                 model_class_name:Union[str, list],
                 model_class_meta_dict:dict,
                 optimizer_class_meta_dict:dict,        # meta_dict or {key_name: meta_dict} / meta_dict: {'name': 'Adam', 'args': {'lr': 0.0001}, model_name_list: []}
                 lr_scheduler_class_meta_dict:dict,
                 loss_class_meta:dict,
                 #train params
                 max_norm_value_for_gradient_clip:float,
                 #train setting
                 total_epoch:int,
                 save_model_every_step:int,
                 seed: float,
                 seed_strict:bool,
                 debug_mode:bool = False
                 ) -> None:
        self.h_params = HParams()
        self.device:torch.device = device

        self.model_class_name:Union[str, list] = model_class_name
        self.model_class_meta_dict:dict = model_class_meta_dict
        self.model:Union[nn.Module, list, dict] = None
        
        self.optimizer_class_meta_dict:dict = optimizer_class_meta_dict
        self.optimizer:torch.optim.Optimizer = None
        self.lr_scheduler_class_meta_dict:dict = lr_scheduler_class_meta_dict
        self.lr_scheduler:torch.optim.lr_scheduler = None

        self.loss_function_dict:dict = dict()
        self.loss_class_meta:dict = loss_class_meta

        self.data_loader_dict:dict = {subset: None for subset in ['train','valid','test']}

        self.seed:int = seed
        self.set_seeds(self.seed, seed_strict)

        self.max_norm_value_for_gradient_clip:float = max_norm_value_for_gradient_clip

        self.current_epoch:int = 1
        self.total_epoch:int = total_epoch
        self.global_step:int = 0
        self.local_step:int = 0
        self.best_valid_metric:dict[str,AverageMeter] = None
        self.best_valid_epoch:int = 0
        self.save_model_every_step:int = save_model_every_step
        
        if debug_mode:
            torch.autograd.set_detect_anomaly(True)

        
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    
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
        
        if prev_best_metric is None:
            return current_metric
        
        if prev_best_metric[self.loss_control.final_loss_name].avg > current_metric[self.loss_control.final_loss_name].avg:
            self.save_module(self.model)
            self.best_valid_epoch = self.current_epoch
            return current_metric
        else:
            return prev_best_metric
        """
        return None
    
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

        log:str = f'Epoch ({train_state.value}): {self.current_epoch:03} ({self.local_step}/{data_size}) global_step: {self.global_step} lr: {self.get_current_lr()}\n'
        
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
    def set_seeds(self, seed:float, strict=False) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if strict:
                torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def init_train(self, dataset_dict=None):
        self.model = self.init_model(self.model_class_name)
        self.optimizer = self.init_optimizer(self.optimizer_class_meta_dict)
        self.lr_scheduler = self.init_lr_scheduler(self.optimizer, self.lr_scheduler_class_meta_dict)
        self.init_loss()
        self.model_to_device()
        
        self.log_writer:LogWriter = LogWriter(model=self.model)
        self.set_data_loader(dataset_dict)
    
    def init_model(self, model_class_name:Union[str, list, dict]) -> None:
        if isinstance(model_class_name, list):
            model = dict()
            for name in model_class_name:
                model[name] = self.init_model(name)
        elif isinstance(model_class_name, dict):
            model = dict()
            for name in model_class_name:
                model[name] = self.init_model(model_class_name[name])
        else:
            model:nn.Module = GetModule.get_model(model_class_name)
        return model
    
    def init_optimizer(self, optimizer_class_meta_dict:dict) -> None:
        optimizer_class_name = optimizer_class_meta_dict.get('name',None)
        if optimizer_class_name is None:
            optimizer = dict()
            for key in optimizer_class_meta_dict:
                optimizer[key] = self.init_optimizer(optimizer_class_meta_dict[key])
        else:
            optimizer_class = getattr(torch.optim, optimizer_class_name)
            model_name_list:list = optimizer_class_meta_dict.get('model_name_list', None)
            if model_name_list is None:
                params = self.model.parameters()
            else:
                params = list()
                for model_name in self.model:
                    if model_name in model_name_list:
                        params += list(self.model[model_name].parameters())

            optimizer_args:dict = {"params": params}
            optimizer_args.update(optimizer_class_meta_dict['args'])
            optimizer_args['lr'] = float(optimizer_args['lr'])
            optimizer = optimizer_class(**optimizer_args)
        return optimizer

    def init_lr_scheduler(self, optimizer, lr_scheduler_class_meta_dict) -> None:
        if isinstance(optimizer, dict):
            lr_scheduler = dict()
            for key in optimizer:
                lr_scheduler[key] = self.init_lr_scheduler(optimizer[key], self.lr_scheduler_class_meta_dict[key])
        else:
            lr_scheduler_name:str = lr_scheduler_class_meta_dict.get('name',None)
            lr_scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler_name)
            lr_scheduler_args:dict = lr_scheduler_class_meta_dict['args']
            lr_scheduler_args.update({'optimizer': optimizer})
            lr_scheduler =  lr_scheduler_class(**lr_scheduler_args)
        return lr_scheduler

    def init_loss(self) -> None:
        for loss_name in self.loss_class_meta:
            loss_class: Type[torch.nn.Module] = getattr(torch.nn, self.loss_class_meta[loss_name]['class_meta']['name']) # loss_name:Literal['L1Loss']
            self.loss_function_dict[loss_name] = loss_class()
    
    def model_to_device(self):
        if self.h_params.resource.multi_gpu:
            from TorchJaekwon.Train.Trainer.Parallel import DataParallelModel, DataParallelCriterion
            self.model = DataParallelModel(self.model)
            self.model.cuda()
            for loss_name in self.loss_control.loss_function_dict:
                self.loss_control.loss_function_dict[loss_name] = DataParallelCriterion(self.loss_control.loss_function_dict[loss_name])
        else:
            for loss_name in self.loss_function_dict:
                self.loss_function_dict[loss_name] = self.loss_function_dict[loss_name].to(self.device)
            if isinstance(self.model_class_name, list):
                for class_name in self.model_class_name: 
                    self.model[class_name] = self.model[class_name].to(self.device)
            elif isinstance(self.model_class_name, dict):
                for type_name in self.model_class_name:
                    for class_name in self.model_class_name[type_name]:
                        self.model[type_name][class_name] = self.model[type_name][class_name].to(self.device)
            else:   
                self.model = self.model.to(self.device)

    def data_dict_to_device(self,data_dict:dict) -> dict:
        for feature_name in data_dict:
            if isinstance(data_dict[feature_name],dict):
                data_dict[feature_name] = self.data_dict_to_device(data_dict[feature_name])
            else:
                data_dict[feature_name] = data_dict[feature_name].float().to(self.device)
        return data_dict
    
    def set_data_loader(self,dataset_dict=None):
        data_loader_getter_class:Type[PytorchDataLoader] = GetModule.get_module_class('./Data/PytorchDataLoader', self.h_params.pytorch_data.class_meta['name'])
        data_loader_getter = data_loader_getter_class(**self.h_params.pytorch_data.class_meta['args'])
        if dataset_dict is not None:
            pytorch_data_loader_config_dict = data_loader_getter.get_pytorch_data_loader_config(dataset_dict)
            self.data_loader_dict = data_loader_getter.get_pytorch_data_loaders_from_config(pytorch_data_loader_config_dict)
        else:
            self.data_loader_dict = data_loader_getter.get_pytorch_data_loaders()
    
    def fit(self) -> None:
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
                self.save_module(self.model, name=f"pretrained_epoch{str(self.current_epoch).zfill(8)}")
            
            self.current_epoch += 1
            self.log_writer.log_every_epoch(model=self.model)

        self.log_writer.print_and_log(f'best_epoch: {self.best_valid_epoch}')
        self.log_writer.print_and_log('Training complete')
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState, metric_range:str = "step"):
        assert metric_range in ["step","epoch"], "metric range should be 'step' or 'epoch'"

        if train_state == TrainState.TRAIN:
            self.set_model_train_valid_mode(self.model, 'train')
        else:
            self.set_model_train_valid_mode(self.model, 'valid')

        try: dataset_size = len(dataloader)
        except: dataset_size = dataloader.dataset.__len__()


        if metric_range == "epoch":
            metric = dict()

        for step,data in enumerate(dataloader):

            if metric_range == "step":
                metric = dict()

            if step >= len(dataloader):
                break

            self.local_step = step
            loss,metric = self.run_step(data,metric,train_state)

            if torch.isnan(loss).any():
                path = os.path.join(self.log_writer.log_path["root"],f'nan_loss_data_{self.global_step}.pkl')
                UtilData.pickle_save(path,data)
                self.save_module(self.model, name=f"nan_loss_step{self.global_step}")
                self.save_checkpoint(f"nan_loss_step{self.global_step}.pth")
                raise ValueError(f'loss is nan at step {self.global_step}')
        
            if train_state == TrainState.TRAIN:
                self.backprop(loss)
                
                if self.local_step % self.h_params.log.log_every_local_step == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                
                self.global_step += 1

                self.lr_scheduler_step(call_state='step')
            
            if train_state == TrainState.TRAIN and self.save_model_every_step is not None and self.global_step % self.save_model_every_step == 0:
                self.save_module(self.model, name=f"step{self.global_step}")
                self.log_current_state()
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.save_module(self.model, name=f"step{self.global_step}")
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)
            self.log_current_state(train_state)

        return metric
    
    def log_current_state(self,train_state:TrainState = None, is_log_media:bool = True) -> None:
        self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------')
        self.log_writer.print_and_log(f'save current state')
        self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------')

        if train_state == TrainState.TRAIN or train_state == None:
            self.save_checkpoint()
            self.save_checkpoint("train_checkpoint_backup.pth")
        if is_log_media:
            with torch.no_grad():
                self.log_media()

        self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------')
        self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------')
    
    def backprop(self,loss):
        if self.max_norm_value_for_gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm_value_for_gradient_clip)

        if getattr(self.h_params.train,'optimizer_step_unit',1) == 1:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            loss.backward()
            if (self.global_step + 1) % self.h_params.train.optimizer_step_unit == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
    
    def set_model_train_valid_mode(self, model, mode: Literal['train','valid']):
        if isinstance(model, dict):
            for model_name in model:
                self.set_model_train_valid_mode(model[model_name], mode)
        else:
            if mode == 'train':
                model.train()
            else:
                model.eval()
                model.zero_grad()
    
    def metric_update(self, metric:Dict[str, AverageMeter], loss_name:str, loss:torch.Tensor, batch_size:int) -> dict:
        if loss_name not in metric:
            metric[loss_name] = AverageMeter()
        metric[loss_name].update(loss.item(),batch_size)
        return metric

    def save_module(self, model, model_name = '', name = 'pretrained_best_epoch'):
        if isinstance(model, dict):
            for model_type in model:
                self.save_module(model[model_type], model_name + f'{model_type}_', name)
        else:
            path = os.path.join(self.log_writer.log_path["root"],f'{model_name}{name}.pth')
            torch.save(model.state_dict() if not self.h_params.resource.multi_gpu else model.module.state_dict(), path)

    def load_module(self,name = 'pretrained_best_epoch'):
        path = os.path.join(self.log_writer.log_path["root"],f'{name}.pth')
        best_model_load = torch.load(path)
        self.model.load_state_dict(best_model_load)
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def lr_scheduler_step(self, call_state:Literal['step','epoch'], args = None):
        if self.lr_scheduler is None:
            return
        if self.h_params.train.scheduler['interval'] == call_state:
            if args is not None:
                self.lr_scheduler.step(**args)
            else:
                self.lr_scheduler.step()
    
    def save_checkpoint(self,save_name:str = 'train_checkpoint.pth'):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'optimizers': self.optimizer.state_dict(),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }

        if self.lr_scheduler is not None:
            train_state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        train_state.update(self.get_model_state_dict(self.model))

        path = os.path.join(self.log_writer.log_path["root"],save_name)
        self.log_writer.print_and_log(save_name)
        torch.save(train_state,path)
    
    def get_model_state_dict(self, model, model_name = '', state_dict = dict()):
        if isinstance(model, dict):
            for model_type in model:
                state_dict.update(self.get_model_state_dict(model[model_type], model_name + f'_{model_type}', state_dict))
        else:
            state_dict[f'model{model_name}'] = model.state_dict() if not self.h_params.resource.multi_gpu else model.module.state_dict()
        return state_dict

    def load_train(self, filename:str) -> None:
        cpt:dict = torch.load(filename,map_location='cpu')
        self.seed = cpt['seed']
        self.set_seeds(self.h_params.train.seed_strict)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']

        self.model = self.model.to(torch.device('cpu'))
        self.model.load_state_dict(cpt['model'])
        self.model = self.model.to(self.device)

        self.optimizer.load_state_dict(cpt['optimizers'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(cpt['lr_scheduler'])
        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']