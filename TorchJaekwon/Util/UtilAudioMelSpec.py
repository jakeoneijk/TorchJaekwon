#type
from typing import Union
from numpy import ndarray
from torch import Tensor
#package
import os
import torch
import numpy as np
import librosa.display
from librosa.filters import mel as librosa_mel_fn
try:
    import matplotlib.pyplot as plt
except:
    print('matplotlib is uninstalled')
#torchjaekwon
from TorchJaekwon.Util.UtilAudioSTFT import UtilAudioSTFT

class UtilAudioMelSpec(UtilAudioSTFT):
    def __init__(self, 
                 nfft: int, 
                 hop_size: int, 
                 sample_rate:int,
                 mel_size:int,
                 frequency_min:float,
                 frequency_max:float):
        super().__init__(nfft, hop_size)

        self.sample_rate:int = sample_rate
        self.mel_size:int = mel_size
        self.frequency_min:float = frequency_min
        self.frequency_max:float = frequency_max

        #[self.mel_size, self.nfft//2 + 1]
        self.mel_basis_np:ndarray = librosa_mel_fn(sr = self.sample_rate,
                                                   n_fft = self.nfft, 
                                                   n_mels = self.mel_size,
                                                   fmin = self.frequency_min, 
                                                   fmax = self.frequency_max)
        self.mel_basis_tensor:Tensor = torch.from_numpy(self.mel_basis_np).float()
    
    @staticmethod
    def get_default_mel_spec_config(sample_rate:int = 16000) -> dict:
        nfft:int = 1024 if sample_rate <= 24000 else 2048
        return {'nfft': nfft, 'hop_size': nfft//4, 'sample_rate': sample_rate, 'mel_size': 80, 'frequency_max': sample_rate//2, 'frequency_min': 0}
    
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
    
    def get_hifigan_mel_spec(self,
                             audio:Union[ndarray,Tensor], #[Batch,Time]
                             return_type:str=['ndarray','Tensor'][1]
                             ) -> Union[ndarray,Tensor]:
        if isinstance(audio,ndarray): audio = torch.FloatTensor(audio)

        if torch.min(audio) < -1.:
            print('min value is ', torch.min(audio))
        if torch.max(audio) > 1.:
            print('max value is ', torch.max(audio))

        spectrogram = self.stft_torch(audio)["mag"]
        mel_spec = self.spec_to_mel_spec(spectrogram)
        log_scale_mel = self.dynamic_range_compression(mel_spec)

        if return_type == 'ndarray':
            return log_scale_mel.cpu().detach().numpy()
        else:
            return log_scale_mel
    
    def save_mel_spec_plot(self,
                           save_path:str, #'*.png'
                           mel_spec:ndarray, #[mel_size, time]
                           dpi:int = 500) -> None:
        assert(os.path.splitext(save_path)[1] == ".png") , "file extension should be '.png'"
        try:
            fig, ax = plt.subplots()        
                
            img =   librosa.display.specshow(
                    mel_spec, 
                    y_axis='mel', 
                    x_axis='time',
                    sr=self.sample_rate, 
                    hop_length=self.hop_size, 
                    fmin=self.frequency_min, 
                    fmax=self.frequency_max, 
                    ax=ax,
                    cmap='viridis')

            ax.set(title='Mel spectrogram display')
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            plt.savefig(save_path,dpi=dpi)
            plt.close()
        except:
            print('there is some problem with matplotlib, so we will use alternative way')
            plt.close()
            self.spec_to_figure(spec=mel_spec,
                                fig_size = None,
                                dpi = dpi,
                                transposed=False,
                                save_path=save_path)
