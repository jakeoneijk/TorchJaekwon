from Train.LossFunction.LossControl import LossControl
from Train.LossFunction.LossEnergy import LossEnergy
from HParams import HParams

class LossControlEnergyLoss(LossControl):
    def __init__(self, h_params: HParams) -> None:
        super().__init__(h_params)
    
    def set_loss_function_dict_additional(self,loss_name:str, lossfunction_name:str) -> None:
        if lossfunction_name == "energy":
            self.loss_function_dict[loss_name] = LossEnergy()