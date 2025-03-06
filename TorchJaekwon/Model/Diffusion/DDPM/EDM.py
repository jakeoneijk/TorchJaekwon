from torch import Tensor, device
from typing import Optional, Union, Literal

import math
import torch

from TorchJaekwon.Util import UtilData, UtilTorch
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM
from TorchJaekwon.Model.Diffusion.External.k_diffusion import sampling as e_sampling
from TorchJaekwon.Model.Diffusion.External.k_diffusion.external import VDenoiser

class EDM(DDPM):
    '''
    Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
    aka k-dissusion / KDiffusion / Karras Diffusion
    '''
    def __init__(self, **kwargs) -> None:
        default_args = dict()
        default_args['time_type'] = 'continuous'
        default_args['timesteps'] = None 
        default_args['betas'] = None
        default_args['beta_schedule_type'] = None
        default_args['beta_arg_dict'] = None
        assert not any([key in kwargs for key in default_args]), f"Can't change the following params: {list(default_args.keys())}"
        kwargs.update(default_args)
        super().__init__(**kwargs)
    
    def q_sample(self, x_start:Tensor, t:Tensor, noise=None) -> Tensor:
        '''
        noisy x sample for forward process
        '''
        alphas, sigmas = self.t_to_alpha_sigma(t)
        alphas = alphas[:, *[ None for _ in range(len(x_start.shape) - 1) ]]
        sigmas = sigmas[:, *[ None for _ in range(len(x_start.shape) - 1) ]]
        
        noise = UtilData.default(noise, lambda: torch.randn_like(x_start))
        return x_start * alphas + noise * sigmas
    
    @torch.no_grad()
    def infer(
        self,
        x_shape:tuple = None,
        cond:Optional[Union[dict,Tensor]] = None,
        is_cond_unpack:bool = False,
        additional_data_dict:dict = None,
        sampler_type:Literal['heun', 'lms', 'dpmpp_2s_ancestral', 'dpm_2', 'dpm_fast', 'dpm_adaptive', 'dpmpp_2m_sde', 'dpmpp_3m_sde'] = 'dpmpp_3m_sde',
        steps:int = 100,
        sigma_min:float = 0.3, #0.5, 
        sigma_max:float = 500, #50, 
        rho:float = 1.0,
    ) -> Tensor:
        if x_shape is None: x_shape = self.get_x_shape(cond)
        model_device:device = UtilTorch.get_model_device(self.model)
        noise:Tensor = torch.randn(x_shape, device = model_device)

        sigmas = e_sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=model_device)
        noise = noise * sigmas[0]

        sampling_args:dict = { 'model': VDenoiser(self.model), 'x': noise, 'disable': False }
        if sampler_type in ['dpm_adaptive', 'dpm_fast']:
            sampling_args['sigma_min'] = sigma_min
            sampling_args['sigma_max'] = sigma_max
            if sampler_type == 'dpm_adaptive':
                sampling_args['rtol'] = 0.01
                sampling_args['atol'] = 0.01
            elif sampler_type == 'dpm_fast':
                sampling_args['n'] = steps
        else:
            sampling_args['sigmas'] = sigmas
        
        sampler_func_name:str = f'sample_{sampler_type}'
        sampler_func = getattr(e_sampling, sampler_func_name)
        x = sampler_func(**sampling_args)

        
        return self.postprocess(x, additional_data_dict = additional_data_dict)

    def alpha_sigma_to_t(self, alpha, sigma):
        """
        Source: https://github.com/crowsonkb/v-diffusion-pytorch
        Returns a timestep, given the scaling factors for the clean image and for the noise.
        """
        return torch.atan2(sigma, alpha) / math.pi * 2

    def t_to_alpha_sigma(self, t):
        """
        Source: https://github.com/crowsonkb/v-diffusion-pytorch
        Returns the scaling factors for the clean image and for the noise, given a timestep."""
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
    
    def get_v(self, x_start, noise, t):
        '''
        Progressive Distillation for Fast Sampling of Diffusion Models
        https://arxiv.org/abs/2202.00512
        '''
        alphas, sigmas = self.t_to_alpha_sigma(t)
        alphas = alphas[:, *[ None for _ in range(len(x_start.shape) - 1) ]]
        sigmas = sigmas[:, *[ None for _ in range(len(x_start.shape) - 1) ]]
        return noise * alphas - x_start * sigmas
