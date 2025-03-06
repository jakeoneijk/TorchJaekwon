from typing import Union, Callable, Literal, Optional, Tuple
from numpy import ndarray
from torch import Tensor, device

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from TorchJaekwon.GetModule import GetModule
from TorchJaekwon.Util import UtilData, UtilTorch

class FlowMatching(nn.Module):
    def __init__(
        self,
        # model
        model_class_name:Optional[str] = None,
        model:Optional[nn.Module] = None,
        # time
        timestep_sampler:Literal['uniform', 'logit_normal'] = 'uniform',
        # loss
        loss_func:Union[nn.Module, Callable, Tuple[str,str]] = F.mse_loss, # if tuple (package name, func name). ex) (torch.nn.functional, mse_loss)
        # classifier free guidance
        unconditional_prob:float = 0, #if unconditional_prob > 0, this model works as classifier free guidance    
        cfg_scale:Optional[float] = None # classifer free guidance scale
    ) -> None:
        super().__init__()
        # model
        if model_class_name is not None:
            self.model = GetModule.get_model(model_name = model_class_name)
        else:
            self.model:nn.Module = model
        # time
        self.timestep_sampler:Literal['uniform', 'logit_normal'] = timestep_sampler
        if self.timestep_sampler == 'uniform':
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        # loss
        self.loss_func:Union[nn.Module, Callable] = loss_func
        # classifier free guidance
        self.unconditional_prob:float = unconditional_prob
        self.cfg_scale:Optional[float] = cfg_scale
    
    def forward(
        self,
        x_start:Tensor,
        cond:Optional[Union[dict,Tensor]] = None,
        is_cond_unpack:bool = False
    ) -> Tensor: # return loss value or sample
        '''
        train diffusion model. 
        return diffusion loss
        '''
        x_start, cond, _ = self.preprocess(x_start, cond)
        batch_size:int = x_start.shape[0] 
        input_device:device = x_start.device
        t:Tensor = self.sample_time(batch_size).to(input_device)
        if self.make_decision(self.unconditional_prob):
            cond:Optional[Union[dict,Tensor]] = self.get_unconditional_condition(cond=cond, condition_device=input_device)
        return self.get_loss(x_start, cond, is_cond_unpack, t)
    
    @torch.no_grad()
    def infer(
        self,
        x_shape:tuple = None,
        cond:Optional[Union[dict,Tensor]] = None,
        is_cond_unpack:bool = False,
        steps:int = 100,
        sigma_max:float = 1
    ) -> Tensor:
        _, cond, additional_data_dict = self.preprocess(None, cond)

        if x_shape is None: x_shape = self.get_x_shape(cond)
        model_device:device = UtilTorch.get_model_device(self.model)
        x:Tensor = torch.randn(x_shape, device = model_device)

        sigma_max = min(sigma_max, 1)

        # discrete_euler, https://github.com/Stability-AI/stable-audio-tools
        # Create the noise schedule
        t = torch.linspace(sigma_max, 0, steps + 1)

        for t_curr, t_prev in tqdm(zip(t[:-1], t[1:])):
                # Broadcast the current timestep to the correct shape
                t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
                dt = t_prev - t_curr  # we solve backwards in our formulation
                x = x + dt * self.apply_model(x = x, t = t_curr_tensor, cond = cond, is_cond_unpack = is_cond_unpack, cfg_scale=self.cfg_scale)

        return self.postprocess(x, additional_data_dict = additional_data_dict)
    
    def get_loss(
        self, 
        x_start:Tensor,
        cond:Optional[Union[dict,Tensor]],
        is_cond_unpack:bool,
        t:Tensor, 
        noise:Optional[Tensor] = None
    ):
        noise:Tensor = UtilData.default(noise, lambda: torch.randn_like(x_start))
        x_noisy:Tensor = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output:Tensor = self.apply_model(x_noisy, t, cond, is_cond_unpack)

        target:Tensor = self.get_target(x_start, noise, t)
        
        if target.shape != model_output.shape: print(f'warning: target shape({target.shape}) and model shape({model_output.shape}) are different')
        return self.loss_func(target, model_output)
    
    def apply_model(
        self,
        x:Tensor,
        t:Tensor,
        cond:Optional[Union[dict,Tensor]],
        is_cond_unpack:bool,
        cfg_scale:Optional[float] = None
    ) -> Tensor:
        if cfg_scale is None or cfg_scale == 1.0:
            if cond is None:
                return self.model(x, t)
            elif is_cond_unpack:
                return self.model(x, t, **cond)
            else:
                return self.model(x, t, cond)
        else:
            model_conditioned_output = self.model(x, t, **cond) if is_cond_unpack else self.model(x, t, cond)
            unconditional_conditioning = self.get_unconditional_condition(cond=cond)
            model_unconditioned_output = self.model(x, t, **unconditional_conditioning) if is_cond_unpack else self.model(x, t, unconditional_conditioning)
            return model_unconditioned_output + cfg_scale * (model_conditioned_output - model_unconditioned_output)
    
    def sample_time(self, batch_size:int) -> Tensor:
        if self.timestep_sampler == 'uniform':
            return self.rng.draw(batch_size)[:, 0]
        elif self.timestep_sampler == 'logit_normal':
            return torch.sigmoid(torch.randn(batch_size))
    
    def q_sample(self, x_start:Tensor, t:Tensor, noise=None) -> Tensor:
        '''
        noisy x sample
        '''
        alphas, sigmas = self.t_to_alpha_sigma(t)
        alphas = alphas[:, *[ None for _ in range(len(x_start.shape) - 1) ]]
        sigmas = sigmas[:, *[ None for _ in range(len(x_start.shape) - 1) ]]
        
        noise = UtilData.default(noise, lambda: torch.randn_like(x_start))
        return x_start * alphas + noise * sigmas
    
    def get_target(self, x_start, noise, t):
        return noise - x_start
    
    def t_to_alpha_sigma(self, t:Tensor) -> Tuple[Tensor,Tensor]:
        return 1-t, t
    
    def make_decision(
        self,
        probability:float #[0,1]
    ) -> bool:
        if probability == 0:
            return False
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False
    
    def get_unconditional_condition(
        self,
        cond:Optional[Union[dict,Tensor]] = None, 
        cond_shape:Optional[tuple] = None,
        condition_device:Optional[device] = None
    ) -> Tensor:
        print('Default Unconditional Condition. You might wanna overwrite this function')
        if cond_shape is None: cond_shape = cond.shape
        if cond is not None and isinstance(cond,Tensor): condition_device = cond.device
        return (-11.4981 + torch.zeros(cond_shape)).to(condition_device)

    def preprocess(self, x_start:Tensor, cond:Optional[Union[dict,Tensor]] = None) -> Tuple[Tensor, Optional[Union[dict,Tensor]], dict]:
        return x_start, cond, None

    def postprocess(self, x:Tensor, additional_data_dict:dict) -> Tensor:
        return x
    
    def get_x_shape(self, cond:Optional[Union[dict,Tensor]] = None):
        return None