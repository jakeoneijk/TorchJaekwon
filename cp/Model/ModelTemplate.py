#####set sys path#####################################
from TorchJaekwon.Util.Util import Util
Util.set_sys_path_to_parent_dir(__file__,1)
#type
#import
import torch
import torch.nn as nn
#torchjaekwon import
#internal import

class ModelTemplate(nn.Module):
    def __init__(self,parameter1,parameter2) -> None:
        super(ModelTemplate,self).__init__()
    
    @staticmethod
    def get_argument_of_this_model() -> dict:
        from HParams import HParams
        h_params = HParams()
        model_argument:dict = getattr(h_params.model,'ModelTemplate',dict())
        model_argument["parameter1"] = h_params.preprocess.parameter1
        model_argument["parameter2"] = h_params.preprocess.parameter2
        return model_argument
    
    @staticmethod
    def get_test_input():
        return torch.rand((4,2,72000))

if __name__ == '__main__':
    test_input = ModelTemplate.get_test_input()
    model = ModelTemplate(**ModelTemplate.get_argument_of_this_model())
    output = model(test_input)
    print('finish')