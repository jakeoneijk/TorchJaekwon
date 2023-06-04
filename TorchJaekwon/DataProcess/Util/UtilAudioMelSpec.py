from numpy import ndarray
from torch import Tensor

import os
import torch
import numpy as np
import librosa.display
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt

from TorchJAEKWON.DataProcess.Util.UtilAudioSTFT import UtilAudioSTFT

class UtilAudioMelSpec(UtilAudioSTFT):
    def __init__(self, nfft: int, hop_size: int, sample_rate:int,mel_size:int,frequency_min:float,frequency_max:float):
        super().__init__(nfft, hop_size)

        self.sample_rate = sample_rate
        self.mel_size = mel_size
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max

        #self.mel_basis_np.shape == (self.mel_size, self.nfft//2 + 1)
        self.mel_basis_np:ndarray = librosa_mel_fn(self.sample_rate, self.nfft, self.mel_size, self.frequency_min, self.frequency_max)
        self.mel_basis_tensor:Tensor = torch.from_numpy(self.mel_basis_np).float()
    
    def spec_to_mel_spec(self,stft_mag):
        if type(stft_mag) == np.ndarray:
            return np.matmul(self.mel_basis_np, stft_mag)
        elif type(stft_mag) == torch.Tensor:
            self.mel_basis_tensor = self.mel_basis_tensor.to(stft_mag.device)
            return torch.matmul(self.mel_basis_tensor, stft_mag)
        else:
            print("spec_to_mel_spec type error")
            exit()
    
    def dynamic_range_compression(self, x, C=1, clip_val=1e-5):
        if type(x) == np.ndarray:
            return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
        elif type(x) == torch.Tensor:
            return torch.log(torch.clamp(x, min=clip_val) * C)
        else:
            print("dynamic_range_compression type error")
            exit()
    
    def get_hifigan_mel_spectrogram_from_audio(self,audio):
        audio = torch.FloatTensor(audio)

        if torch.min(audio) < -1.:
            print('min value is ', torch.min(audio))
        if torch.max(audio) > 1.:
            print('max value is ', torch.max(audio))

        spectrogram = self.stft_torch(audio)["mag"]
        mel_spec = self.spec_to_mel_spec(spectrogram)
        log_scale_mel = self.dynamic_range_compression(mel_spec)

        return log_scale_mel
    
    def save_mel_spec_plot(self,save_path:str,mel_spec:ndarray):
        assert(os.path.splitext(save_path)[1] == ".png") , "file extension should be '.png'"

        fig, ax = plt.subplots()        
            
        img =   librosa.display.specshow(
                mel_spec, 
                y_axis='mel', 
                x_axis='time',
                sr=self.sample_rate, 
                hop_length=self.hop_size, 
                fmin=self.frequency_min, 
                fmax=self.frequency_max, 
                ax=ax)

        ax.set(title='Mel spectrogram display')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.savefig(save_path,dpi=1000)