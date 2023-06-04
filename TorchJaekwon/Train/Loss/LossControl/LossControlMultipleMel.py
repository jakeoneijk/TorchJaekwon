from HParams import HParams
import torch

from Train.Loss.LossControl.LossControl import LossControl

class LossControlMultipleMel(LossControl):
    def __init__(self, h_params: HParams) -> None:
        super().__init__(h_params)
    
    def set_loss_function_dict(self):
        for i in range(self.h_params.train.volume_reduce_iteration):
            self.loss_function_dict[f"mel_{i}"] = torch.nn.L1Loss()
    
    def calculate_total_loss_by_loss_meta_dict(self, pred_dict:dict, target_dict:dict):
        loss_dict = dict()
        for loss_name in pred_dict:

            assert pred_dict[loss_name].shape == target_dict[loss_name].shape, f"Loss shape problem: {pred_dict[loss_name].shape} and {target_dict[loss_name].shape}"
            loss_dict[loss_name] = self.loss_function_dict[loss_name](pred_dict[loss_name],target_dict[loss_name])

            if self.final_loss_name in loss_dict:
                loss_dict[self.final_loss_name] = loss_dict[self.final_loss_name] + loss_dict[loss_name]
            else:
                loss_dict[self.final_loss_name] = loss_dict[loss_name]
        return loss_dict
    