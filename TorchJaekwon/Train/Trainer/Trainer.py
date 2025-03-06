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
try: from ema_pytorch import EMA
except: print("ema_pytorch is not installed")
#torchjaekwon import
from TorchJaekwon.GetModule import GetModule
from TorchJaekwon.Data.PytorchDataLoader.PytorchDataLoader import PytorchDataLoader
from TorchJaekwon.Train.LogWriter.LogWriter import LogWriter
from TorchJaekwon.Util import Util, UtilData, UtilTorch

from TorchJaekwon.Train.AverageMeter import AverageMeter
#internal import

@unique
class TrainState(Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"
 
class Trainer():
    def __init__(
        self,
        # data
        data_class_meta_dict:dict,
        # model
        model_class_name:Union[str, list],
        model_ckpt_path:str = None,
        # loss
        loss_class_meta:dict = None,
        # optimizer
        optimizer_class_meta_dict:dict = None,        # meta_dict or {key_name: meta_dict} / meta_dict: {'name': 'Adam', 'args': {'lr': 0.0001}, model_name_list: []}
        optimizer_step_unit:int = 1,
        lr_scheduler_class_meta_dict:dict = None,
        lr_scheduler_interval:Literal['step','epoch'] = 'step',
        max_norm_value_for_gradient_clip:float = None,
        use_ema:bool = False,
        # train paremeters
        total_step:int = np.inf,
        total_epoch:int = int(1e20),
        seed:int = (int)(torch.cuda.initial_seed() / (2**32)),
        seed_strict:bool = False,
        # logging
        save_model_step_interval:int = None,
        save_model_epoch_interval:int = 1,
        log_step_interval:int = 100,
        start_logging_epoch:bool = 0,
        # resource
        device:torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        multi_gpu:bool = False,
        # additional
        check_evalstep_first:bool = False,
        debug_mode:bool = False,
        use_torch_compile:bool = True
    ) -> None:
        # data
        self.data_class_meta_dict:dict = data_class_meta_dict
        self.data_loader_dict:dict = {subset: None for subset in ['train','valid','test']}

        # model
        self.model_class_name:Union[str, list] = model_class_name
        self.model:Union[nn.Module, list, dict] = None
        self.model_ckpt_path:str = model_ckpt_path

        # loss
        self.loss_class_meta:dict = loss_class_meta
        self.loss_function_dict:dict = dict()

        # optimizer
        self.optimizer_class_meta_dict:dict = optimizer_class_meta_dict
        self.optimizer_step_unit:int = optimizer_step_unit
        self.lr_scheduler_class_meta_dict:dict = lr_scheduler_class_meta_dict
        self.lr_scheduler_interval:Literal['step','epoch'] = lr_scheduler_interval
        self.max_norm_value_for_gradient_clip:float = max_norm_value_for_gradient_clip
        self.use_ema:bool = use_ema
        self.model_ema:nn.Module = None
        self.optimizer:torch.optim.Optimizer = None
        self.lr_scheduler:torch.optim.lr_scheduler = None
        
        # train paremeters
        self.total_step:int = total_step
        self.total_epoch:int = total_epoch
        self.seed:int = seed
        self.seed_strict:bool = seed_strict
        self.set_seeds(self.seed, self.seed_strict)
        self.current_epoch:int = 1
        self.global_step:int = 0
        self.local_step:int = 0

        # logging
        self.save_model_step_interval:int = save_model_step_interval
        self.save_model_epoch_interval:int = save_model_epoch_interval
        self.log_step_interval:int = log_step_interval
        self.start_logging_epoch:bool = start_logging_epoch

        # resource
        self.device:torch.device = device
        self.multi_gpu:bool = multi_gpu

        # additional
        self.check_evalstep_first:bool = check_evalstep_first
        self.debug_mode = debug_mode
        self.use_torch_compile = use_torch_compile
        if debug_mode:
            Util.print("debug mode is on", msg_type='warning')
            torch.autograd.set_detect_anomaly(True)
        else:
            Util.print("debug mode is off. \n  - [off] torch.autograd.set_detect_anomaly", msg_type='info')
            if self.use_torch_compile: 
                Util.print("\n  - [on] torch.compile", msg_type='info')
            else:
                Util.print("\n  - [off] torch.compile", msg_type='warning')

        # evaluation
        self.best_valid_metric:dict[str,AverageMeter] = None
        self.best_valid_epoch:int = 0
        
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
        4. put the loss in metric (append)
            metric = self.update_metric(metric, loss_dict, data.shape[0])
        5. return loss_dict["total_loss"],metric
        """
        raise NotImplementedError

    @torch.no_grad()
    def log_media(self) -> None:
        pass

    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def save_best_model( self, prev_best_metric, current_metric ):
        return None
    
    def update_metric(self, metric:Dict[str,AverageMeter], loss_dict:Dict[str,torch.Tensor], batch_size:int) -> dict:
        for loss_name in loss_dict:
            if loss_name not in metric:
                metric[loss_name] = AverageMeter()
            metric[loss_name].update(loss_dict[loss_name].item(), batch_size)
        return metric
    
    def log_metric(
        self, 
        metrics:Dict[str,AverageMeter],
        data_size: int,
        train_state=TrainState.TRAIN,
        x_axis:Literal['global_step', 'epoch'] = 'global_step'
    ) -> None:
        """
        log and visualizer log
        """
        if x_axis == 'global_step':
            x_axis_name:str = "global_step"
            x_axis_value:int = self.global_step
        else:
            x_axis_name:str = "epoch"
            x_axis_value:int = self.current_epoch

        log:str = f'Epoch ({train_state.value}): {self.current_epoch:03} ({self.local_step}/{data_size}) global_step: {self.global_step} lr: {self.get_current_lr(self.optimizer)}\n'
        
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
        if self.model_ckpt_path is not None:
            ckpt:dict = torch.load(self.model_ckpt_path, map_location='cpu')
            self.model = self.load_state_dict(self.model, ckpt, is_model=True)
        if self.use_ema: self.init_model_ema()
        self.optimizer = self.init_optimizer(self.optimizer_class_meta_dict)
        if self.lr_scheduler_class_meta_dict is not None:
            self.lr_scheduler = self.init_lr_scheduler(self.optimizer, self.lr_scheduler_class_meta_dict)
        self.init_loss()
        self.model_to_device(self.model)
        
        self.log_writer:LogWriter = LogWriter(model=self.model)
        self.set_data_loader(dataset_dict)
    
    def init_model_ema(self) -> None:
        self.model_ema = EMA(
            self.model,
            beta=0.9999,
            power=3/4,
            update_every=1,
            update_after_step=1,
            include_online_model=False
        )
        self.model_ema = self.model_ema.to(self.device)

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
            if not self.debug_mode and self.use_torch_compile:
                model = torch.compile(model)
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
                params = self.get_params(self.model, model_name_list)

            optimizer_args:dict = {"params": params}
            optimizer_args.update(optimizer_class_meta_dict['args'])
            optimizer_args['lr'] = float(optimizer_args['lr'])
            optimizer = optimizer_class(**optimizer_args)
        return optimizer
    
    def get_params(
        self, 
        model:dict, 
        model_name_list:list
    ) -> dict:
        params = list()
        for model_name in model:
            if isinstance(model[model_name], nn.Module):
                if model_name in model_name_list:
                    params += list(model[model_name].parameters())
            else:
                #model[model_name] is dict
                params += self.get_params(model[model_name], model_name_list)
        return params

    def init_lr_scheduler(self, optimizer, lr_scheduler_class_meta_dict) -> None:
        if isinstance(optimizer, dict):
            lr_scheduler = dict()
            for key in optimizer:
                lr_scheduler[key] = self.init_lr_scheduler(optimizer[key], self.lr_scheduler_class_meta_dict[key])
        else:
            lr_scheduler_name:str = lr_scheduler_class_meta_dict.get('name',None)
            lr_scheduler_class = getattr( torch.optim.lr_scheduler, lr_scheduler_name, None )
            if lr_scheduler_class is None:
                lr_scheduler_class = GetModule.get_module_class(class_type = 'lr_scheduler', module_name=lr_scheduler_name)
            lr_scheduler_args:dict = lr_scheduler_class_meta_dict['args']
            lr_scheduler_args.update({'optimizer': optimizer})
            lr_scheduler =  lr_scheduler_class(**lr_scheduler_args)
            if hasattr(lr_scheduler, 'interval') and getattr(lr_scheduler, 'interval', None) != self.lr_scheduler_interval:
                Util.print(
                    text = f'lr_scheduler interval ({self.lr_scheduler_interval}) is not same as interval of {lr_scheduler_name} ({lr_scheduler.interval}).', 
                    msg_type='warning'
                )
        return lr_scheduler

    def init_loss(self) -> None:
        for loss_name in self.loss_class_meta:
            loss_class: Type[torch.nn.Module] = getattr(torch.nn, self.loss_class_meta[loss_name]['class_meta']['name']) # loss_name:Literal['L1Loss']
            self.loss_function_dict[loss_name] = loss_class()
    
    def model_to_device(self, model:Union[nn.Module, dict], device = None) -> None:
        if isinstance(model, dict):
            for model_name in model:
                self.model_to_device(model[model_name])
        else:
            if device is None:
                model = model.to(self.device)
            else:
                model = model.to(device)

    def data_dict_to_device(self,data_dict:dict) -> dict:
        for feature_name in data_dict:
            if isinstance(data_dict[feature_name],dict):
                data_dict[feature_name] = self.data_dict_to_device(data_dict[feature_name])
            else:
                if data_dict[feature_name].dtype in [torch.int64, torch.int32]:
                    data_dict[feature_name] = data_dict[feature_name].to(self.device)
                else:
                    data_dict[feature_name] = data_dict[feature_name].float().to(self.device)
        return data_dict
    
    def set_data_loader(self,dataset_dict=None):
        data_loader_getter_class:Type[PytorchDataLoader] = GetModule.get_module_class(
            root_path = './Data/PytorchDataLoader', 
            module_name = self.data_class_meta_dict['name']
        )
        data_loader_getter = data_loader_getter_class(**self.data_class_meta_dict['args'])
        if dataset_dict is not None:
            pytorch_data_loader_config_dict = data_loader_getter.get_pytorch_data_loader_config(dataset_dict)
            self.data_loader_dict = data_loader_getter.get_pytorch_data_loaders_from_config(pytorch_data_loader_config_dict)
        else:
            self.data_loader_dict = data_loader_getter.get_pytorch_data_loaders()
    
    def fit(self) -> None:
        if self.check_evalstep_first:
            print("check evaluation step first whether there is no error")
            with torch.no_grad():
                valid_metric = self.run_epoch(self.data_loader_dict['valid'],TrainState.VALIDATE, metric_range = "epoch")
                self.log_current_state()
                
        for _ in range(self.current_epoch, self.total_epoch):
            self.log_writer.print_and_log(f'----------------------- Start epoch : {self.current_epoch} / {self.total_epoch} -----------------------')
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
                self.lr_scheduler_step(call_state='epoch') #args=valid_metric)
            
            self.best_valid_metric = self.save_best_model(self.best_valid_metric, valid_metric)

            if self.current_epoch > self.start_logging_epoch and self.current_epoch % self.save_model_epoch_interval == 0:
                self.save_module(self.model, name=f"step{self.global_step}_epoch{self.current_epoch}")
                self.log_current_state()
            
            self.current_epoch += 1
            self.log_writer.log_every_epoch(model=self.model)

            if self.global_step >= self.total_step:
                break

        self.log_writer.print_and_log(f'best_epoch: {self.best_valid_epoch}')
        self.log_writer.print_and_log('Training complete')
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState, metric_range:str = "step") -> dict:
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

            if isinstance(loss, torch.Tensor) and torch.isnan(loss).any():
                path = os.path.join(self.log_writer.log_path["root"],f'nan_loss_data_{self.global_step}.pkl')
                UtilData.pickle_save(path,data)
                self.save_module(self.model, name=f"nan_loss_step{self.global_step}")
                self.save_checkpoint(f"nan_loss_step{self.global_step}.pth")
                raise ValueError(f'loss is nan at step {self.global_step}')
        
            if train_state == TrainState.TRAIN:
                self.backprop(loss)
                self.lr_scheduler_step(call_state='step')
                
                if self.global_step % self.log_step_interval == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                
                if self.save_model_step_interval is not None and self.global_step % self.save_model_step_interval == 0 and not self.global_step == 0:
                    self.save_module(self.model, name=f"step{self.global_step}")
                    self.log_current_state()

                self.global_step += 1
                if self.global_step >= self.total_step:
                    return metric
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)
            
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

        if self.optimizer_step_unit == 1:
            self.on_before_zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            loss.backward()
            if (self.global_step + 1) % self.optimizer_step_unit == 0:
                self.optimizer.step()
                self.on_before_zero_grad()
                self.optimizer.zero_grad()
    
    def on_before_zero_grad(self) -> None:
        if self.model_ema is not None:
            self.model_ema.update()
    
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
            if self.multi_gpu:
                state_dict = model.module.state_dict()
            elif self.model_ema is not None:
                state_dict = self.model_ema.ema_model.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, path)

    def load_module(self,name = 'pretrained_best_epoch'):
        path = os.path.join(self.log_writer.log_path["root"],f'{name}.pth')
        best_model_load = torch.load(path)
        self.model.load_state_dict(best_model_load)
    
    def get_current_lr(self, optimizer:Union[ dict, torch.optim.Optimizer]):
        if isinstance(optimizer, dict):
            return self.get_current_lr(optimizer[list(optimizer.keys())[0]])
        else:
            return optimizer.param_groups[0]['lr']
    
    def lr_scheduler_step(self, call_state:Literal['step','epoch'], args = None):
        if self.lr_scheduler is None:
            return
        if self.lr_scheduler_interval == call_state:
            if args is not None:
                if isinstance(self.lr_scheduler, dict):
                    for key in self.lr_scheduler:
                        self.lr_scheduler[key].step(**args)
                else:
                    self.lr_scheduler.step(**args)
            else:
                if isinstance(self.lr_scheduler, dict):
                    for key in self.lr_scheduler:
                        self.lr_scheduler[key].step()
                else:
                    self.lr_scheduler.step()
    
    def save_checkpoint(self,save_name:str = 'train_checkpoint.pth'):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'model': self.get_state_dict(self.model),
            'optimizers': self.get_state_dict(self.optimizer),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }

        if self.lr_scheduler is not None:
            train_state['lr_scheduler'] = self.get_state_dict(self.lr_scheduler)
        
        if self.use_ema:
            train_state['model_ema'] = self.get_state_dict(self.model_ema)

        path = os.path.join(self.log_writer.log_path["root"],save_name)
        self.log_writer.print_and_log(save_name)
        torch.save(train_state,path)
    
    def get_state_dict(self, module:Union[dict, nn.Module]) -> Union[dict, nn.Module]:
        if hasattr(module, 'state_dict'):
            return module.state_dict()
        elif isinstance(module, dict):
            state_dict = dict()
            for key in module:
                state_dict[key] = self.get_state_dict(module[key])
            return state_dict
        else:
            raise ValueError(f'Cannot get state_dict from {module}')
    
    def load_state_dict(self, module:Union[dict, nn.Module], state_dict:dict, is_model:bool = False) -> Union[dict, nn.Module]:
        if hasattr(module, 'load_state_dict'):
            if is_model:
                module = UtilTorch.load_model(module, state_dict)
            else:
                module.load_state_dict(state_dict)
            return module
        elif isinstance(module, dict):
            for key in module:
                module[key] = self.load_state_dict(module[key], state_dict[key])
            return module
        else:
            raise ValueError(f'Cannot load state_dict to {module}')

    def load_train(self, filename:str) -> None:
        self.log_writer.print_and_log(f'load train from {filename}')
        cpt:dict = torch.load(filename,map_location='cpu')
        self.seed = cpt['seed']
        self.set_seeds(self.seed_strict)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']

        self.model_to_device(self.model, torch.device('cpu'))
        self.model = self.load_state_dict(self.model, cpt['model'], is_model=True)  
        self.model_to_device(self.model)
        if self.use_ema:
            self.model_ema = self.load_state_dict(self.model_ema, cpt['model_ema'])
            self.model_to_device(self.model_ema)

        self.optimizer = self.load_state_dict(self.optimizer, cpt['optimizers'])
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.load_state_dict(self.lr_scheduler, cpt['lr_scheduler'])
        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']