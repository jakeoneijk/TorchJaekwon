from typing import Dict, Optional
from numpy import ndarray

import os
import psutil
import time
import torch.nn as nn
from datetime import datetime
from datetime import timedelta
try: import wandb
except: print('Didnt import following packages: wandb')
try: from tensorboardX import SummaryWriter
except: print('Didnt import following packages: tensorboardX')

from TorchJaekwon.Util.UtilAudioSTFT import UtilAudioSTFT
from TorchJaekwon.Util.UtilTorch import UtilTorch

from HParams import HParams

class LogWriter():
    def __init__(self,
                 model:nn.Module
                 )->None:
        self.h_params:HParams = HParams()
        self.visualizer_type:str = self.h_params.log.visualizer_type #["tensorboard","wandb"]

        self.experiment_start_time:float = time.time()
        self.experiment_name = "[" +datetime.now().strftime('%y%m%d-%H%M%S') + "] " + self.h_params.mode.config_name if self.h_params.log.use_currenttime_on_experiment_name else self.h_params.mode.config_name
        
        self.log_path:dict[str,str] = {"root":"","console":"","visualizer":""}
        self.set_log_path()
        self.log_write_init(model=model)

        if self.visualizer_type == 'wandb':
            wandb.init(project=self.h_params.log.project_name)
            wandb.config = {"learning_rate": self.h_params.train.lr, "epochs": self.h_params.train.epoch, "batch_size": self.h_params.train.batch_size }
            wandb.watch(model[list(model.keys())[0]] if isinstance(model, dict) else model[0] if isinstance(model, list) else model)
            wandb.run.name = self.experiment_name
            wandb.run.save()
        elif self.visualizer_type == 'tensorboard':
            self.tensorboard_writer = SummaryWriter(log_dir=self.log_path["visualizer"])
        else:
            print('visualizer should be either wandb or tensorboard')
            exit()
    
    def get_time_took(self) -> str:
        time_took_second:int = int(time.time() - self.experiment_start_time)
        time_took:str = str(timedelta(seconds=time_took_second))
        return time_took
    
    def set_log_path(self):
        if self.h_params.mode.train == "resume":
            self.log_path["root"] = self.h_params.mode.resume_path
        else:
            self.log_path["root"] = os.path.join(self.h_params.log.class_root_dir,self.experiment_name)
        self.log_path["console"] = self.log_path["root"]+ "/log.txt"
        self.log_path["visualizer"] = os.path.join(self.log_path["root"],"tb")

        os.makedirs(self.log_path["visualizer"],exist_ok=True)
        
    def print_and_log(self,log_message:str) -> None:
        log_message_with_time_took:str = f"{log_message} ({self.get_time_took()} took)"
        print(log_message_with_time_took)
        self.log_write(log_message_with_time_took)
    
    def log_write_init(self,model:nn.Module) -> None:
        file = open(self.log_path["console"],'w')
        file.write("========================================="+'\n')
        file.write(f'pid: {os.getpid()} / parent_pid: {psutil.Process(os.getpid()).ppid()} \n')
        file.write("========================================="+'\n')
        if isinstance(model,dict):
            for model_name in model:
                file.write(f'''Model {model_name} Total parameters: {format(UtilTorch.get_param_num(model[model_name])['total'], ',d')}'''+'\n')
                file.write(f'''Model {model_name} Trainable parameters: {format(UtilTorch.get_param_num(model[model_name])['trainable'], ',d')}'''+'\n')
        elif isinstance(model,list):
            for model_idx in range(len(model)):
                file.write(f'''Model {model_idx} Total parameters: {format(UtilTorch.get_param_num(model[model_idx])['total'], ',d')}'''+'\n')
                file.write(f'''Model {model_idx} Trainable parameters: {format(UtilTorch.get_param_num(model[model_idx])['trainable'], ',d')}'''+'\n')
        else:
            file.write(f'''Model Total parameters: {format(UtilTorch.get_param_num(model)['total'], ',d')}'''+'\n')
            file.write(f'''Model Trainable parameters: {format(UtilTorch.get_param_num(model)['trainable'], ',d')}'''+'\n')
        file.write("========================================="+'\n')
        file.write("Epoch :" + str(self.h_params.train.epoch)+'\n')
        file.write("lr :" + str(self.h_params.train.lr)+'\n')
        file.write("Batch :" + str(self.h_params.train.batch_size)+'\n')
        file.write("========================================="+'\n')
        file.close()

    def log_write(self,log_message:str)->None:
        file = open(self.log_path["console"],'a')
        file.write(log_message+'\n')
        file.close()

    def visualizer_log(
        self,
        x_axis_name:str,
        x_axis_value:float,
        y_axis_name:str,
        y_axis_value:float) -> None:

        if self.visualizer_type == 'tensorboard':
            self.tensorboard_writer.add_scalar(y_axis_name,y_axis_value,x_axis_value)
        else:
            wandb.log({y_axis_name: y_axis_value, x_axis_name: x_axis_value})
    
    def plot_audio(self, 
                   name:str, #test case name, you could make structure by using /. ex) 'taskcase_1/test_set_1'
                   audio_dict:Dict[str,ndarray], #{'audio name': 1d audio array}.
                   global_step:int,
                   sample_rate:int = 16000,
                   is_plot_spec:bool = False,
                   is_plot_mel:bool = True,
                   mel_spec_args:Optional[dict] = None
                   ) -> None:
        self.plot_wav(name = name + '_audio', audio_dict = audio_dict, sample_rate=sample_rate, global_step=global_step)
        if is_plot_mel:
            from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec
            if mel_spec_args is None:
                mel_spec_args = UtilAudioMelSpec.get_default_mel_spec_config(sample_rate=sample_rate)
            mel_spec_util = UtilAudioMelSpec(**mel_spec_args)
            mel_dict = dict()
            for audio_name in audio_dict:
                mel_dict[audio_name] = mel_spec_util.get_hifigan_mel_spec(audio=audio_dict[audio_name],return_type='ndarray')
            self.plot_spec(name = name + '_mel_spec', spec_dict = mel_dict)
        
    
    def plot_wav(self, 
                 name:str, #test case name, you could make structure by using /. ex) 'audio/test_set_1'
                 audio_dict:Dict[str,ndarray], #{'audio name': 1d audio array},
                 sample_rate:int,
                 global_step:int
                 ) -> None:
        if self.visualizer_type == 'tensorboard':
            for audio_name in audio_dict:
                self.tensorboard_writer.add_audio(f'{name}/{audio_name}', audio_dict[audio_name], sample_rate=sample_rate, global_step=global_step)
        else:
            wandb_audio_list = list()
            for audio_name in audio_dict:
                wandb_audio_list.append(wandb.Audio(audio_dict[audio_name], caption=audio_name,sample_rate=sample_rate))
            wandb.log({name: wandb_audio_list})
    
    def plot_spec(self, 
                  name:str, #test case name, you could make structure by using /. ex) 'mel/test_set_1'
                  spec_dict:Dict[str,ndarray], #{'name': 2d array},
                  vmin=-6.0, 
                  vmax=1.5,
                  transposed=False, 
                  global_step=0):
        if self.visualizer_type == 'tensorboard':
            for audio_name in spec_dict:
                figure = UtilAudioSTFT.spec_to_figure(spec_dict[audio_name], vmin=vmin, vmax=vmax,transposed=transposed)
                self.tensorboard_writer.add_figure(f'{name}/{audio_name}',figure,global_step=global_step)
        else:
            wandb_mel_list = list()
            for audio_name in spec_dict:
                UtilAudioSTFT.spec_to_figure(spec_dict[audio_name], vmin=vmin, vmax=vmax,transposed=transposed,save_path=f'''{self.log_path['root']}/temp_img_{audio_name}.png''')
                wandb_mel_list.append(wandb.Image(f'''{self.log_path['root']}/temp_img_{audio_name}.png''', caption=audio_name))
            wandb.log({name: wandb_mel_list})
    
    def log_every_epoch(self,model:nn.Module):
        pass