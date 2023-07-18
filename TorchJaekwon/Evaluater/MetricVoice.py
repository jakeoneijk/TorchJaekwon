#type
from torch import Tensor
from numpy import ndarray

import numpy as np
try:
    import pyworld as pw
    import pysptk
    import torch
    from pesq import pesq
    from fastdtw import fastdtw
    from TorchJaekwon.DataProcess.Util.UtilAudioMelSpec import UtilAudioMelSpec
    from TorchJaekwon.DataProcess.Util.UtilAudio import UtilAudio
    from skimage.metrics import structural_similarity as ssim
except:
    pass

class MetricVoice:
    def __init__(self,sample_rate:int = 16000) -> None:
        self.util_mel = UtilAudioMelSpec(nfft = 512, 
                                         hop_size = 256, 
                                         sample_rate = sample_rate, 
                                         mel_size = 80,
                                         frequency_min = 0,
                                         frequency_max = float(sample_rate // 2))
        '''
        self.mel_44k = MelScale(n_mels=128, sample_rate=44100, n_stft=1025)
        self.mel_16k = MelScale(n_mels=80, sample_rate=16000, n_stft=372)
        '''
    def get_spec_metrics_from_audio(self,
                                   source, #linear scale spectrogram [time]
                                   target):
        source_spec_dict = self.get_spec_dict_of_audio(source)
        target_spec_dict = self.get_spec_dict_of_audio(target)

        metric_dict = dict()
        for spec_name in source_spec_dict:
            metric_dict[f'lsd_{spec_name}'] = MetricVoice.get_lsd_from_spec(source_spec_dict[spec_name],target_spec_dict[spec_name])
            metric_dict[f'ssim_{spec_name}'] = float(ssim(source_spec_dict[spec_name],target_spec_dict[spec_name],win_size=7))
        
        linear_spec_name = list(source_spec_dict.keys())
        for spec_name in linear_spec_name:
            source_spec_dict[f'{spec_name}_log'] = np.log10(np.clip(source_spec_dict[spec_name], a_min=1e-8, a_max=None))
            target_spec_dict[f'{spec_name}_log'] = np.log10(np.clip(target_spec_dict[spec_name], a_min=1e-8, a_max=None))
        
        for spec_name in source_spec_dict:
            metric_dict[f'sispnr_{spec_name}'] = MetricVoice.get_sispnr(torch.from_numpy(source_spec_dict[spec_name]),torch.from_numpy(target_spec_dict[spec_name]))
        
        return metric_dict
        
    
    def get_lsd_from_audio(self,
                           source, #linear scale spectrogram [time]
                           target):
        source_spec_dict = self.get_spec_dict_of_audio(source)
        target_spec_dict = self.get_spec_dict_of_audio(target)
        lsd_dict = dict()
        for spec_name in source_spec_dict:
            lsd_dict[spec_name] = MetricVoice.get_lsd_from_spec(source_spec_dict[spec_name],target_spec_dict[spec_name])
        return lsd_dict
    
    def get_spec_dict_of_audio(self,audio):
        spectrogram_mag = self.util_mel.stft_torch(audio)['mag'].float()
        mel_spec = self.util_mel.spec_to_mel_spec(spectrogram_mag)
        return {'spec_mag':spectrogram_mag.squeeze().detach().cpu().numpy(), 'mel': mel_spec.squeeze().detach().cpu().numpy()}

    @staticmethod
    def get_lsd_from_spec(source, #linear scale spectrogram [freq, time]
                          target,
                          eps = 1e-12):
        # in non-log scale
        lsd = np.log10((target**2/((source + eps)**2)) + eps)**2 #torch.log10((target**2/((source + eps)**2)) + eps)**2
        lsd = np.mean(np.mean(lsd,axis=1)**0.5,axis=0) #torch.mean(torch.mean(lsd,dim=3)**0.5,dim=2)
        return float(lsd)
    
    @staticmethod
    def get_si_sdr(source, target):
        alpha = np.dot(target, source)/np.linalg.norm(source)**2   
        sdr = 10*np.log10(np.linalg.norm(alpha*source)**2/np.linalg.norm(
            alpha*source - target)**2)
        return sdr
    
    @staticmethod
    def get_pesq(source:ndarray, #[time]
                 target:ndarray, #[time]
                 sample_rate:int = [8000,16000][1],
                 band:str = ['wide-band','narrow-band'][0]):
        assert (sample_rate in [8000,16000]), f'sample rate must be either 8000 or 16000. current sample rate {sample_rate}'
        if (sample_rate == 16000 and band == 'narrow-band'): print('Warning: narrowband (nb) mode only when sampling rate is 8000Hz')
        if band == 'wide-band':
            return pesq(sample_rate, target, source, 'wb')
        else:
            return pesq(sample_rate, target, source, 'nb')
        
    @staticmethod
    def get_mcd(source:ndarray, #[time]
                target:ndarray, #[time]
                sample_rate:int, 
                frame_period=5):
        cost_function = MetricVoice.dB_distance
        mgc_source = MetricVoice.get_mgc(source, sample_rate, frame_period)
        mgc_target = MetricVoice.get_mgc(target, sample_rate, frame_period)

        length = min(mgc_source.shape[0], mgc_target.shape[0])
        mgc_source = mgc_source[:length]
        mgc_target = mgc_target[:length]

        mcd, _ = fastdtw(mgc_source[..., 1:], mgc_target[..., 1:], dist=cost_function)
        mcd = mcd/length

        return float(mcd), length
    
    @staticmethod
    def get_mgc(audio, sample_rate, frame_period, fft_size=512, mcep_size=60, alpha=0.65):
        if isinstance(audio, Tensor):
            if audio.ndim > 1:
                audio = audio[0]

            audio = audio.numpy()

        _, sp, _ = pw.wav2world(
            audio.astype(np.double), fs=sample_rate, frame_period=frame_period, fft_size=fft_size)
        mgc = pysptk.sptk.mcep(
            sp, order=mcep_size, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

        return mgc
    
    @staticmethod
    def dB_distance(source, target):
        dB_const = 10.0/np.log(10.0)*np.sqrt(2.0)
        distance = source - target

        return dB_const*np.sqrt(np.inner(distance, distance))
    
    @staticmethod
    def get_sispnr(source, target, eps = 1e-12):
        # scale_invariant_spectrogram_to_noise_ratio
        # in log scale
        output, target = UtilAudio.energy_unify(source, target)
        noise = output - target
        # print(pow_p_norm(target) , pow_p_norm(noise), pow_p_norm(target) / (pow_p_norm(noise) + EPS))
        sp_loss = 10 * torch.log10((UtilAudio.pow_p_norm(target) / (UtilAudio.pow_p_norm(noise) + eps) + eps))
        return float(sp_loss)
    
    
'''

    
    
    def get_sdr_torchmetrics(self,pred_audio:Union[Tensor,ndarray], target_audio:Union[Tensor,ndarray]) -> dict:
        result_dict = dict()
        audio_spec_tensor_dict:dict = self.get_audio_and_spec_tensor_pred_target_dict(pred_audio, target_audio)
        for data_type in audio_spec_tensor_dict["pred"]:
            sdr = SignalDistortionRatio()
            result_dict[f"sdr_torchmetrics_{data_type}"] = float(sdr(audio_spec_tensor_dict["pred"][data_type].clone(),audio_spec_tensor_dict["target"][data_type].clone()))

        return result_dict

    
    
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
'''