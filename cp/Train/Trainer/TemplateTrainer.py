import numpy as np

from TorchJAEKWON.Train.Trainer.Trainer import Trainer, TrainState
from TorchJAEKWON.DataProcess.Process.Process import Process

class TemplateTrainer(Trainer):

    def __init__(self):
        super().__init__()
        if self.h_params.process.name is not None and self.h_params.process.name != "":
            self.data_processor:Process = self.get_module.get_module("process",self.h_params.process.name, {"h_params":self.h_params},arg_unpack=True)
        else:
            self.data_processor = None
    
    def get_train_data_dict_and_name_dict(self,data,data_subset):
        dataset_config = self.h_params.pytorch_data.dataloader["train"]['dataset']

        train_data_name_dict = dict()
        train_data_name_dict["input_name"] = dataset_config["train_source_name_dict"]["input"]
        train_data_name_dict["target_name"] = dataset_config["train_source_name_dict"]["target"]

        for data_name in data:
            data[data_name] = data[data_name].float().to(self.h_params.resource.device)

        if self.data_processor is not None:
            train_data_dict = self.data_processor.preprocess_training_data(data,additional_dict=train_data_name_dict)
        else:
            train_data_dict = data
        
        for train_data_name in train_data_dict:
            if type(train_data_dict[train_data_name]) == dict:
                for sub_train_data_name in train_data_dict[train_data_name]:
                    train_data_dict[train_data_name][sub_train_data_name] = train_data_dict[train_data_name][sub_train_data_name]

        return train_data_name_dict, train_data_dict

    
    def run_step(self,data,metric,train_state):
        """
        run 1 step
        1. get data
        2. use model
        3. calculate loss
        4. put the loss in metric (append)
        return loss,metric
        """
        #import torch
        #print(f"max is {torch.max(self.model.module.impulse_response_reverb)}, min is {torch.min(self.model.module.impulse_response_reverb)}")
        
        train_data_name_dict, train_data_dict = self.get_train_data_dict_and_name_dict(data=data,data_subset=train_state.value)

        batch_size = train_data_dict[train_data_name_dict["input_name"]].size(0)

        pred = self.model(train_data_dict[train_data_name_dict["input_name"]])

        current_loss_dict = self.loss_control.calculate_total_loss_by_loss_meta_dict(pred_dict=pred, target_dict=train_data_dict)

        for loss_name in current_loss_dict:
            metric[loss_name].update(current_loss_dict[loss_name].item(),batch_size)
        
        if TrainState.VALIDATE == train_state:
            self.log_validation(pred)

        return current_loss_dict["total_loss"],metric
    
    def log_validation(self,pred:dict) -> None:
        batch_size:int = pred['waveform'].shape[0]
        audio_dict = dict()
        for i in range(batch_size):
            audio_dict[f"{i}_audio"] = np.mean(pred['waveform'][i].detach().cpu().numpy(),axis=0)
        self.log_writer.visualizer_log_audio_dict(log_name="pred_dry_audio",audio_dict=audio_dict)
    
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

        prev_metric_min = 50
        current_metric_min = 50

        for metric_name in prev_best_metric:
            if metric_name in self.h_params.train.best_model_exception:
                continue
            prev_metric_min = min(prev_metric_min,prev_best_metric[metric_name].avg)
            current_metric_min = min(current_metric_min,current_metric[metric_name].avg)

        
        if prev_metric_min > current_metric_min:
            self.save_module()
            self.best_valid_epoch = self.current_epoch
            return current_metric
        else:
            return prev_best_metric