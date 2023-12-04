import torch
import torch.nn.functional as F

class Util:
    @staticmethod
    def batch_convolution(x, f, pad_both_sides=True):
        """
        Do batch-elementwise convolution between a batch of signals `x` and batch of filters `f`
        x: (batch_size x channels x signal_length) size tensor
        f: (batch_size x channels x filter_length) size tensor
        pad_both_sides: Whether to zero-pad x on left and right or only on left (Default: True)
        """
        batch_size = x.shape[0]
        f = torch.flip(f, (2,))
        if pad_both_sides:
            x = F.pad(x, (f.shape[2]//2, f.shape[2]-f.shape[2]//2-1))
        else:
            x = F.pad(x, (f.shape[2]-1, 0))
        #TODO: This assumes single-channel audio, fine for now 
        return F.conv1d(x.view(1, batch_size, -1), f, groups=batch_size).view(batch_size, 1, -1)
    
    @staticmethod
    def augment(sample, rir=None, noise=None, eq_model=None, low_cut_model=None, rate=16000, nsr_range=[-30,-5], normalize=True, eps=1e-6):
        sample = Util.perturb_silence(sample, eps=eps)
        clean_sample = torch.clone(sample)
        if not noise is None:
            nsr_target = ((nsr_range[1] - nsr_range[0])*torch.rand(noise.shape[0]) + nsr_range[0]).to(noise)
            sample = apply_noise(sample, noise, nsr_target)
        if not rir is None:
            sample = apply_reverb(sample, rir, None, rate=rate)
        if not eq_model is None:
            sample = eq_model(sample)
        if not low_cut_model is None:
            sample = low_cut_model(sample)
        if normalize:
            sample = 0.95*sample/sample.abs().max(dim=2, keepdim=True)[0]

        return clean_sample, sample
    
    @staticmethod
    def perturb_silence(sample, eps=1e-6):
        """
        Some samples have periods of silence which can cause numerical issues when taking log-spectrograms. Add a little noise
        """
        return sample + eps*torch.randn_like(sample)