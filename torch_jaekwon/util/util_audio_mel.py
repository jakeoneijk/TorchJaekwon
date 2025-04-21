#type
from typing import Union
from numpy import ndarray
from torch import Tensor
#package
import os
import torch
import numpy as np
import librosa
try: import librosa.display
except: print('can not import librosa display')
from librosa.filters import mel as librosa_mel_fn
try: import matplotlib.pyplot as plt
except: print('matplotlib is uninstalled')
#torchjaekwon
from .util_audio_stft import UtilAudioSTFT
from .util_torch import UtilTorch

class UtilAudioMelSpec(UtilAudioSTFT):
    def __init__(
        self, 
        nfft: int, 
        hop_size: int, 
        sample_rate:int,
        mel_size:int,
        frequency_min:float,
        frequency_max:float
    ) -> None:
        super().__init__(nfft, hop_size)

        self.sample_rate:int = sample_rate
        self.mel_size:int = mel_size
        self.frequency_min:float = frequency_min
        self.frequency_max:float = frequency_max if frequency_max is not None else sample_rate//2

        #[self.mel_size, self.nfft//2 + 1]
        self.mel_basis_np:ndarray = librosa_mel_fn(
            sr = self.sample_rate,
            n_fft = self.nfft, 
            n_mels = self.mel_size,
            fmin = self.frequency_min, 
            fmax = self.frequency_max
        )
        self.mel_basis_tensor:Tensor = torch.from_numpy(self.mel_basis_np).float()
        self.mel_frequncies = librosa.mel_frequencies(
            n_mels = self.mel_size,
            fmin = self.frequency_min, 
            fmax = self.frequency_max
        )
    
    @staticmethod
    def get_default_config(sample_rate:int = 16000) -> dict:
        nfft:int = 1024 if sample_rate <= 24000 else 2048
        mel_size:int = 80 if sample_rate <= 24000 else 128
        return {'nfft': nfft, 'hop_size': nfft//4, 'sample_rate': sample_rate, 'mel_size': mel_size, 'frequency_max': sample_rate//2, 'frequency_min': 0}
    
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
    
    def get_hifigan_mel_spec(
        self,
        audio:Union[ndarray,Tensor], #[Batch,Time]
        return_type:str=['ndarray','Tensor'][1]
    ) -> Union[ndarray,Tensor]:
        if isinstance(audio,ndarray): audio = torch.FloatTensor(audio)
        while len(audio.shape) < 2: audio = audio.unsqueeze(0)

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
    
    @staticmethod
    def plot(
        save_path:str, #'*.png'
        mel_spec:ndarray, #[mel_size, time]
        fig_size:tuple=(12,4),
        dpi:int = 150,
        hop_size:int = None,
        sr:int = None,
        line_dict:dict = dict(), # {'feature_name': {'value': 1d_array[time], 'color':str , 'scale':bool}}
        h_line_dict:dict = dict(), # {'feature_name': List[{'start':int, 'end':int, value:float}]}
        text_dict:dict = dict(), #{ 'feature_name': List[{ x:int, y:int, value_dict:List[{'value':str}] }] }
    ) -> None:
        assert(os.path.splitext(save_path)[1] == ".png") , "file extension should be '.png'"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # set plt
        plt.rc('font', size=4)
        plt.rc('legend', labelcolor='white')
        plt.grid(False)
        if isinstance(mel_spec, Tensor):
            mel_spec = UtilTorch.to_np(mel_spec)
        COLOR_DICT = {
            'blue': 'blue',
            'darkblue':'darkblue',
            'red': 'red',
            'darkred': 'darkred',
            'purple': '#9370DB',
            'darkpurple': '#4B0082',
        }
        color_list = ['darkblue', 'blue', 'darkmagenta', 'darkorange', 'darkgreen', 'darkblue', 'darkred', 'darkcyan', 'darkviolet', 'darkgoldenrod', 'darkolivegreen', 'darkslategray']
        _, axes = plt.subplots(1, 1, figsize=fig_size, sharex=True)
        axes.grid(True, axis='x', linestyle='--', color='black', linewidth=0.5)
        # set x axis
        axes.set_xlim([0, mel_spec.shape[-1]])
        time_axis = np.arange(mel_spec.shape[-1]) * hop_size / sr
        axes.set_xticks(np.linspace(0, mel_spec.shape[-1], num=10))
        axes.set_xticklabels(np.round(np.linspace(0, time_axis[-1], num=10), 2))
        # set y axis
        axes.set_ylim([0, mel_spec.shape[0]])
        axes.imshow(mel_spec, origin='lower', aspect='auto', cmap='viridis')
        axes.set_yticks([])
        # plot line
        for key, value in line_dict.items():
            y = value['value']
            if y is None: continue
            if value.get('scale', False):
                y = y * mel_spec.shape[0]
            color = COLOR_DICT.get(value.get('color', ''), color_list.pop())
            axes.plot(y, color = color, linewidth=1, label=key)
        # plot horizontal line
        for key, value in h_line_dict.items():
            for plot_dict in value:
                plt.plot((plot_dict['start'], plot_dict['end']), (plot_dict['value'], plot_dict['value']), 'k', linewidth=1)
        # plot text
        text_y_offset = 4
        for key, value in text_dict.items():
            for plot_dict in value:
                for i, value_dict in enumerate(plot_dict['value_dict']):
                    axes.text(
                        x=plot_dict['x'],
                        y=max(plot_dict['y'] - text_y_offset * (i + 1), 0),
                        s=value_dict['value'],
                        color='black',
                        horizontalalignment='center',
                        verticalalignment='center'
                    )
        axes.legend()
        plt.tight_layout()
        plt.savefig(save_path,dpi=dpi)
        plt.close()
    
    def f0_to_melbin(
        self, 
        f0:Tensor # 1d f0 tensor
    ) -> Tensor:
        mel_frequencies = torch.FloatTensor(self.mel_frequncies).repeat(f0.shape[0]).reshape(f0.shape[0],-1).to(f0.device)
        mel_frequencies[((mel_frequencies - f0.unsqueeze(-1)) < 0)] = np.inf
        all_inf_value = torch.all(torch.isinf(mel_frequencies), dim = 1)
        mel_frequencies[all_inf_value,-1] = 0
        return torch.argmin(mel_frequencies, dim=1)
