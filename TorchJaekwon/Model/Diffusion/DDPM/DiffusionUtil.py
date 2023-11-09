from torch import Tensor,device

import torch

class DiffusionUtil:
    @staticmethod
    def extract(array:Tensor, t, x_shape):
        batch_size, *_ = t.shape
        out = array.gather(dim = -1, index = t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def noise_like(shape:tuple, device:device, repeat:bool = False):
        repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
        noise = lambda: torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()
