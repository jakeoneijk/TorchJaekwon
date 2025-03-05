from torch import Tensor

import math
import torch

from TorchJaekwon.Util.UtilData import UtilData
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM

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