import torch.nn as nn

import os
import time
from datetime import datetime
from datetime import timedelta

import wandb


from HParams import HParams

class LogWriter():
    def __init__(self)->None:
        self.h_params:HParams = HParams()
        self.experiment_start_time:float = time.time()
        self.experiment_name = "[" +datetime.now().strftime('%y%m%d-%H%M%S') + "] " + self.h_params.mode.config_name if self.h_params.log.use_currenttime_on_experiment_name else self.h_params.mode.config_name
        
        self.log_path:dict[str,str] = {"root":"","console":"","visualizer":""}
        self.set_log_path()
        '''
        
        elif self.h_params.log.visualizer_type == "wandb":
            wandb.init(project=self.h_params.log.project_name)
            wandb.config = {"learning_rate": self.h_params.train.lr, "epochs": self.h_params.train.epoch, "batch_size": self.h_params.train.batch_size }
            wandb.watch(config['model'])
            wandb.run.name = self.experiment_name
            wandb.run.save()
        '''
    
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
        
    def print_and_log(self,log_message:str,global_step:int) -> None:
        log_message_with_time_took:str = f"{log_message} ({self.get_time_took()} took)"
        print(log_message_with_time_took)
        self.log_write(log_message_with_time_took,global_step)

    def log_write(self,log_message:str,global_step:int)->None:
        if global_step == 0:
            file = open(self.log_path["console"],'w')
            file.write("========================================="+'\n')
            file.write("Epoch :" + str(self.h_params.train.epoch)+'\n')
            file.write("lr :" + str(self.h_params.train.lr)+'\n')
            file.write("Batch :" + str(self.h_params.train.batch_size)+'\n')
            file.write("========================================="+'\n')
            file.close()

        file = open(self.log_path["console"],'a')
        file.write(log_message+'\n')
        file.close()
    
    def visualizer_log(
        self,
        x_axis_name:str,
        x_axis_value:float,
        y_axis_name:str,
        y_axis_value:float) -> None:

        if self.h_params.log.visualizer_type == "tensorboard":
            self.tensorboard_writer.add_scalar(y_axis_name,y_axis_value,x_axis_value)
        elif self.h_params.log.visualizer_type == "wandb":
            wandb.log({y_axis_name: y_axis_value, x_axis_name: x_axis_value})
    
    def visualizer_log_audio_dict(self,log_name:str,audio_dict:dict) -> None:
        if self.h_params.log.visualizer_type == "wandb":
            wandb_audio_list = list()
            for audio_name in audio_dict:
                wandb_audio_list.append(wandb.Audio(audio_dict[audio_name], caption=audio_name,sample_rate=self.h_params.preprocess.sample_rate))
            wandb.log({log_name: wandb_audio_list})
    
    def log_every_epoch(self,model:nn.Module):
        pass