from copy import deepcopy
from typing import Union
from numpy import ndarray
from torch import Tensor, from_numpy

import torch
from torchmetrics.audio import SignalDistortionRatio
import numpy as np
from HParams import HParams
from DataProcess.Util.UtilAudioMelSpec import UtilAudioMelSpec
import pyworld as pw
import pysptk
from fastdtw import fastdtw

class MetricVoice:
    def __init__(self, h_params:HParams) -> None:
        self.h_params:HParams = h_params
        self.util = UtilAudioMelSpec(h_params)
        self.eps:float = 1e-12
    
    
    def get_sispnr(self,pred_audio:Tensor, target_audio:Tensor) -> dict:
        result_dict = dict()
        audio_spec_tensor_dict:dict = self.get_audio_and_spec_tensor_pred_target_dict(pred_audio, target_audio)
        for data_type in audio_spec_tensor_dict["pred"]:
            if "audio" in data_type:
                continue
            result_dict[f"sispnr_{data_type}"] = self.sispnr_scale_invariant_spectrogram_to_noise_ratio(audio_spec_tensor_dict["pred"][data_type].clone(),audio_spec_tensor_dict["target"][data_type].clone())

        return result_dict
    
    def get_sdr_torchmetrics(self,pred_audio:Union[Tensor,ndarray], target_audio:Union[Tensor,ndarray]) -> dict:
        result_dict = dict()
        audio_spec_tensor_dict:dict = self.get_audio_and_spec_tensor_pred_target_dict(pred_audio, target_audio)
        for data_type in audio_spec_tensor_dict["pred"]:
            sdr = SignalDistortionRatio()
            result_dict[f"sdr_torchmetrics_{data_type}"] = float(sdr(audio_spec_tensor_dict["pred"][data_type].clone(),audio_spec_tensor_dict["target"][data_type].clone()))

        return result_dict

    def get_audio_and_spec_tensor_pred_target_dict(self,pred_audio:Union[Tensor,ndarray], target_audio:Union[Tensor,ndarray]) -> dict:
        audio_spec_tensor_dict:dict = {"pred":dict(),"target":dict()}
        audio_spec_tensor_dict["pred"]["audio"] = pred_audio.clone().detach() if isinstance(pred_audio, Tensor) else torch.from_numpy(pred_audio)
        audio_spec_tensor_dict["target"]["audio"] = target_audio.clone().detach() if isinstance(target_audio, Tensor) else torch.from_numpy(target_audio)

        for data_type in audio_spec_tensor_dict:
            audio_spec_tensor_dict[data_type]["spec_linear_scale"] = self.util.stft_torch(audio_spec_tensor_dict[data_type]["audio"])["mag"]
            audio_spec_tensor_dict[data_type]["mel_spec_linear_scale"] = self.util.spec_to_mel_spec(audio_spec_tensor_dict[data_type]["spec_linear_scale"])
            audio_spec_tensor_dict[data_type]["spec_log_scale"] = self.log_scale(audio_spec_tensor_dict[data_type]["spec_linear_scale"].clone().detach())
            audio_spec_tensor_dict[data_type]["mel_spec_log_scale"] = self.log_scale(audio_spec_tensor_dict[data_type]["mel_spec_linear_scale"].clone().detach())
        
        return audio_spec_tensor_dict
    
    def sispnr_scale_invariant_spectrogram_to_noise_ratio(self,pred:Tensor, target:Tensor)->float:
        # in log scale
        output, target = self.energy_unify(pred, target)
        noise = output - target
        # print(pow_p_norm(target) , pow_p_norm(noise), pow_p_norm(target) / (pow_p_norm(noise) + EPS))
        sp_loss = 10 * torch.log10((self.pow_p_norm(target) / (self.pow_p_norm(noise) + self.eps) + self.eps))
        return float(torch.sum(sp_loss))#float(torch.sum(sp_loss) / sp_loss.size()[0])
    
    def log_scale(self, input:Tensor) -> Tensor:
        return torch.log10(torch.clip(input, min=1e-8))

    def energy_unify(self,estimated, original):
        inner_product:Tensor = self.pow_norm(estimated, original)
        denominator:Tensor = self.pow_p_norm(original) + self.eps
        target:Tensor = (inner_product * original) / denominator
        return estimated, target
    
    def pow_norm(self,s1, s2):
        """
        shape = list(s1.size())
        dimension = []
        for i in range(len(shape)):
            if(i == 0 or i == 1):continue
            dimension.append(i)
        return torch.sum(s1 * s2, dim=dimension, keepdim=True)
        """
        return torch.sum(s1 * s2)
    
    def pow_p_norm(self,signal:Tensor):
        """Compute 2 Norm
        shape = list(signal.size())
        dimension = []
        for i in range(len(shape)):
            if(i == 0):continue
            dimension.append(i)
        return torch.pow(torch.norm(signal, p=2, dim=dimension, keepdim=True), 2)
        """
        return torch.pow(torch.norm(signal, p=2), 2)

def get_mgc(audio, sample_rate, frame_period, fft_size=512, mcep_size=60, alpha=0.65):
    if isinstance(audio, torch.Tensor):
        if audio.ndim > 1:
            audio = audio[0]

        audio = audio.numpy()

    _, sp, _ = pw.wav2world(
        audio.astype(np.double), fs=sample_rate, frame_period=frame_period, fft_size=fft_size)
    mgc = pysptk.sptk.mcep(
        sp, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    return mgc


def dB_distance(source, target):
    dB_const = 10.0/np.log(10.0)*np.sqrt(2.0)
    distance = source - target

    return dB_const*np.sqrt(np.inner(distance, distance))


def get_mcd(source, target, sample_rate, frame_period=5, cost_function=dB_distance):
    mgc_source = get_mgc(source, sample_rate, frame_period)
    mgc_target = get_mgc(target, sample_rate, frame_period)

    length = min(mgc_source.shape[0], mgc_target.shape[0])
    mgc_source = mgc_source[:length]
    mgc_target = mgc_target[:length]

    mcd, _ = fastdtw(mgc_source[..., 1:], mgc_target[..., 1:], dist=cost_function)
    mcd = mcd/length

    return mcd, length


def get_f0(audio, sample_rate, frame_period=5, method='dio'):
    if isinstance(audio, torch.Tensor):
        if audio.ndim > 1:
            audio = audio[0]

        audio = audio.numpy()

    hop_size = int(frame_period*sample_rate/1000)
    if method == 'dio':
        f0, _ = pw.dio(audio.astype(np.double), sample_rate, frame_period=frame_period)
    elif method == 'harvest':
        f0, _ = pw.harvest(audio.astype(np.double), sample_rate, frame_period=frame_period)
    elif method == 'swipe':
        f0 = pysptk.sptk.swipe(audio.astype(np.double), sample_rate, hopsize=hop_size)
    elif method == 'rapt':
        f0 = pysptk.sptk.rapt(audio.astype(np.double), sample_rate, hopsize=hop_size)
    else:
        raise ValueError(f'No such f0 extract method, {method}.')

    f0 = torch.from_numpy(f0)
    vuv = 1*(f0 != 0.0)

    return f0, vuv


def get_f0_rmse(source, target, sample_rate, frame_period=5, method='dio'):
    length = min(source.shape[-1], target.shape[-1])

    source_f0, source_v = get_f0(source[...,:length], sample_rate, frame_period, method)
    target_f0, target_v = get_f0(target[...,:length], sample_rate, frame_period, method)

    source_uv = 1 - source_v
    target_uv = 1 - target_v
    tp_mask = source_v*target_v

    length = tp_mask.sum().item()

    f0_rmse = 1200.0*torch.abs(torch.log2(target_f0 + target_uv) - torch.log2(source_f0 + source_uv))
    f0_rmse = tp_mask*f0_rmse
    f0_rmse = f0_rmse.sum()/length

    return f0_rmse.item(), length