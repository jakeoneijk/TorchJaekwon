from typing import Union, Callable, Literal, Optional, Tuple
from numpy import ndarray
from torch import Tensor, device

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ....get_module import GetModule
from ....util import UtilData, UtilTorch
from .time_sampler import TimeSampler
from .diffusion_util import DiffusionUtil
from .beta_schedule import BetaSchedule

class DDPM(nn.Module):
    def __init__(
        self,
        # model
        model_class_meta:Optional[dict] = None, #{name:[file_name, class_name], args: {}}
        model:Optional[nn.Module] = None,
        model_output_type:Literal['noise', 'x_start', 'v_prediction'] = 'noise',
        # time
        time_type:Literal['continuous', 'discrete'] = 'discrete',
        timesteps:int = 1000,
        timestep_sampler:Literal['uniform', 'logit_normal'] = 'uniform',
        # betas schedule
        betas: Optional[ndarray] = None, 
        beta_schedule_type:Literal['linear','cosine'] = 'cosine',
        beta_arg_dict:dict = dict(),
        # loss
        loss_func:Union[nn.Module, Callable, Tuple[str,str]] = F.mse_loss, # if tuple (package name, func name). ex) (torch.nn.functional, mse_loss)
        # classifier free guidance
        unconditional_prob:float = 0, #if unconditional_prob > 0, this model works as classifier free guidance    
        cfg_scale:Optional[float] = None, # classifer free guidance scale
        cfg_calc_type:Literal['batch', 'sequential'] = 'batch'
    ) -> None:
        super().__init__()
        # model
        if model_class_meta is not None:
            model_class = GetModule.get_module_class(class_type = 'model', module_name = model_class_meta['name'])
            self.model = model_class(**model_class_meta['args'])
        else:
            self.model:nn.Module = model
        self.model_output_type:Literal['noise', 'x_start', 'v_prediction'] = model_output_type
        # time
        self.time_sampler = TimeSampler(time_type = time_type, sampler_type = timestep_sampler, timesteps = timesteps)
        # betas schedule
        if any(x is not None for x in (betas, beta_schedule_type, beta_arg_dict)):
            self.set_noise_schedule(betas=betas, beta_schedule_type=beta_schedule_type, beta_arg_dict=beta_arg_dict, timesteps=timesteps)
        # loss
        self.loss_func:Union[nn.Module, Callable] = loss_func
        # classifier free guidance
        self.unconditional_prob:float = unconditional_prob
        self.cfg_scale:Optional[float] = cfg_scale
        self.cfg_calc_type:Literal['batch', 'sequential'] = cfg_calc_type
    
    def set_noise_schedule(
        self,
        betas: Optional[ndarray] = None, 
        beta_schedule_type:Literal['linear','cosine'] = 'linear',
        beta_arg_dict:dict = dict(),
        timesteps:int = 1000,
    ) -> None:
        if betas is None:
            beta_arg_dict.update({'timesteps':timesteps})
            betas = getattr(BetaSchedule,beta_schedule_type)(**beta_arg_dict)
        
        alphas:ndarray = 1. - betas
        alphas_cumprod:ndarray = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev:ndarray = np.append(1., alphas_cumprod[:-1])

        self.betas:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'betas', value = betas)
        self.alphas_cumprod:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'alphas_cumprod', value = alphas_cumprod)
        self.alphas_cumprod_prev:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'alphas_cumprod_prev', value = alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'sqrt_alphas_cumprod', value = np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'sqrt_one_minus_alphas_cumprod', value = np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'log_one_minus_alphas_cumprod', value = np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'sqrt_recip_alphas_cumprod', value = np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'sqrt_recipm1_alphas_cumprod', value = np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'posterior_variance', value = posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'posterior_log_variance_clipped', value = np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'posterior_mean_coef1', value = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2:Tensor = UtilTorch.register_buffer(model = self, variable_name = 'posterior_mean_coef2', value = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    
    def forward(
        self,
        x_start:Optional[Tensor] = None,
        x_shape:Optional[tuple] = None,
        cond:Optional[Union[dict,Tensor]] = None,
        is_cond_unpack:bool = True,
        stage: Literal['train', 'infer'] = 'train'
    ) -> Tensor: # return loss value or sample
        '''
        train diffusion model. 
        return diffusion loss
        '''
        x_start, cond, additional_data_dict = self.preprocess(x_start, cond)
        if stage == 'train' and x_start is not None:
            if x_shape is None: x_shape = x_start.shape
            batch_size:int = x_shape[0] 
            input_device:device = x_start.device
            t:Tensor = self.time_sampler.sample(batch_size).to(input_device)
            if self.unconditional_prob > 0:
                uncond_dict:dict = self.get_unconditional_condition(cond=cond, condition_device=input_device)
                for cond_name, uncond in uncond_dict.items():
                    dropout_mask = torch.bernoulli(torch.full((uncond.shape[0], *[1 for _ in range(len(uncond.shape) - 1)]), self.unconditional_prob, device=input_device)).to(torch.bool)
                    cond[cond_name] = torch.where(dropout_mask, uncond, cond[cond_name])
            return self.p_losses(x_start, cond, is_cond_unpack, t)
        else:
            return self.infer(x_shape = x_shape, cond = cond, is_cond_unpack = is_cond_unpack, additional_data_dict = additional_data_dict)
    
    def p_losses(
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

        if self.model_output_type == 'x_start':
            target:Tensor = x_start
        elif self.model_output_type == 'noise':
            target:Tensor = noise
        elif self.model_output_type == 'v_prediction':
            target:Tensor = self.get_v(x_start, noise, t)
        else:
            print(f'''model output type is {self.model_output_type}. It should be in [x_start, noise]''')
            raise NotImplementedError()
        if target.shape != model_output.shape: print(f'warning: target shape({target.shape}) and model shape({model_output.shape}) are different')
        return self.loss_func(target, model_output)
    
    def get_v(self, x_start, noise, t):
        '''
        Progressive Distillation for Fast Sampling of Diffusion Models
        https://arxiv.org/abs/2202.00512
        '''
        return (
            DiffusionUtil.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - DiffusionUtil.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def q_sample(self, x_start:Tensor, t:Tensor, noise=None) -> Tensor:
        '''
        noisy x sample for forward process
        '''
        noise = UtilData.default(noise, lambda: torch.randn_like(x_start))
        return (
            DiffusionUtil.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            DiffusionUtil.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = DiffusionUtil.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = DiffusionUtil.extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = DiffusionUtil.extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    @torch.no_grad()
    def infer(
        self,
        x_shape:tuple,
        cond:Optional[Union[dict,Tensor]],
        is_cond_unpack:bool,
        additional_data_dict:dict
    ) -> Tensor:
        if x_shape is None: x_shape = self.get_x_shape(cond)
        model_device:device = UtilTorch.get_model_device(self.model)
        x:Tensor = torch.randn(x_shape, device = model_device)
        for i in tqdm(reversed(range(0, self.time_sampler.timesteps)), desc='sample time step', total=self.time_sampler.timesteps):
            x = self.p_sample(x = x, t = torch.full((x_shape[0],), i, device= model_device, dtype=torch.long), cond = cond, is_cond_unpack = is_cond_unpack)
        
        return self.postprocess(x, additional_data_dict = additional_data_dict)
    
    @torch.no_grad()
    def p_sample(
        self,
        x:Tensor,
        t:Tensor,
        cond:Optional[Union[dict,Tensor]],
        is_cond_unpack:bool,
        clip_denoised:bool = False, # dangerous if True
        repeat_noise:bool = False
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, cond = cond, is_cond_unpack = is_cond_unpack, clip_denoised = clip_denoised)
        noise = DiffusionUtil.noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def p_mean_variance(
        self,
        x:Tensor,
        t:Tensor,
        cond:Optional[Union[dict,Tensor]],
        is_cond_unpack:bool,
        clip_denoised: bool
    ) -> Tuple[Tensor]:
        
        model_output:Tensor = self.apply_model(x, t, cond, is_cond_unpack, cfg_scale=self.cfg_scale)
        if self.model_output_type == "noise":
            x_recon = self.predict_x_start_from_noise(x, t=t, noise=model_output)
        elif self.model_output_type == 'x_start':
            x_recon = model_output
        elif self.model_output_type == 'v_prediction':
            x_recon = self.predict_x_start_from_v(x, t=t, v=model_output)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def predict_x_start_from_noise(self, x_t, t, noise):
        return (
            DiffusionUtil.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            DiffusionUtil.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_x_start_from_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
            DiffusionUtil.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - DiffusionUtil.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_noise_from_v(self, x_t, t, v):
        return (
            DiffusionUtil.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + DiffusionUtil.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            * x_t
        )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            DiffusionUtil.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            DiffusionUtil.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = DiffusionUtil.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = DiffusionUtil.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def preprocess(self, x_start:Tensor, cond:Optional[Union[dict,Tensor]] = None) -> Tuple[Tensor, Optional[Union[dict,Tensor]], dict]:
        return x_start, cond, None

    def postprocess(self, x:Tensor, additional_data_dict:dict) -> Tensor:
        return x

    def apply_model(
        self,
        x:Tensor,
        t:Tensor,
        cond:Optional[Union[dict,Tensor]],
        cfg_scale:Optional[float] = None
    ) -> Tensor:
        if cfg_scale is None or cfg_scale == 1.0:
            if cond is None:
                return self.model(x, t)
            else:
                return self.model(x, t, **cond)
        else:
            uncond_dict:dict = self.get_unconditional_condition(cond=cond)
            uncond:dict = {key: uncond_dict.get(key, cond[key]) for key in cond}
            if self.cfg_calc_type == 'sequential':
                output_cond = self.model(x, t, **cond)
                output_uncond = self.model(x, t, **uncond)
            else:
                cfg_x = torch.cat([x, x], dim=0)
                cfg_t = torch.cat([t, t], dim=0)
                cfg_cond = {key: torch.cat([cond[key], uncond[key]], dim=0) for key in cond}
                output_cond_uncond = self.model(cfg_x, cfg_t, **cfg_cond)
                output_cond, output_uncond = torch.chunk(output_cond_uncond, 2, dim=0)

            output_cfg = output_uncond + cfg_scale * (output_cond - output_uncond)

            return output_cfg
    
    def get_unconditional_condition(
        self,
        cond:Optional[Union[dict,Tensor]] = None, 
        cond_shape:Optional[tuple] = None,
        condition_device:Optional[device] = None
    ) -> Tensor:
        print('Default Unconditional Condition. You might wanna overwrite this function')
        if cond_shape is None: cond_shape = cond.shape
        if cond is not None and isinstance(cond,Tensor): condition_device = cond.device
        return {'audio': (-11.4981 + torch.zeros(cond_shape)).to(condition_device)}
    
    def get_x_shape(self, cond:Optional[Union[dict,Tensor]] = None):
        return None

if __name__ == '__main__':
    DDPM(model = 'debug')

    
    
        
