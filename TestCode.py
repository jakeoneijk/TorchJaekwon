import torch
from torch import nn
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM

if __name__ == '__main__':
    DDPM(nn.Conv2d(1,3,1))(torch.zeros(8,256,128))