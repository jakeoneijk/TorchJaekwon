from typing import Optional, Literal, Union, Final
from numpy import ndarray
try: from torch import Tensor
except: print('import error: torch')

import numpy as np
import soundfile as sf
import librosa

try: import torch 
except: print('import error: torch')
try: import torchaudio
except: print('import error: torch')
try: from pydub import AudioSegment  
except: print('import error: pydub')

DATA_TYPE_MIN_MAX_DICT:Final[dict] = {'float32':(-1,1), 'float64':(-1,1), 'int16':(-2**15, 2**15-1), 'int32':(-2**31,2**31-1)}

class UtilAudio:
    @staticmethod
    def change_dtype(audio:ndarray,
                     current_dtype:Literal['float32', 'float64', 'int16', 'int32'],
                     target_dtype:Literal['float32', 'float64', 'int16', 'int32']
                     ) -> ndarray:
        audio = np.clip(audio, a_min = DATA_TYPE_MIN_MAX_DICT[current_dtype][0], a_max = DATA_TYPE_MIN_MAX_DICT[current_dtype][1])
        audio = audio / DATA_TYPE_MIN_MAX_DICT[current_dtype][1]
        audio = (audio * DATA_TYPE_MIN_MAX_DICT[target_dtype][1])
        audio = audio.astype(getattr(np,target_dtype))
        return audio
    
    @staticmethod
    def resample_audio(audio,origin_sr,target_sr,resample_type = "kaiser_fast"):
        if(origin_sr == target_sr): return audio
        print(f"resample audio {origin_sr} to {target_sr}")
        return librosa.core.resample(audio, orig_sr=origin_sr, target_sr=target_sr, res_type=resample_type)
    
    @staticmethod
    def read(audio_path:str,
             sample_rate:Optional[int] = None,
             mono:Optional[bool] = None,
             module_name:Literal['soundfile','librosa', 'torchaudio'] = 'soundfile',
             return_type:Union[ndarray, Tensor] = ndarray
            ) -> Union[ndarray, Tensor]: #[shape=(channel, num_samples) or (num_samples)]
        
        if module_name == "soundfile":
            audio_data, original_samplerate = sf.read(audio_path)
            audio_data = audio_data.T

            if sample_rate is not None and sample_rate != original_samplerate:
                print(f"resample audio {original_samplerate} to {sample_rate}")
                audio_data = UtilAudio.resample_audio(audio_data,original_samplerate,sample_rate)

        elif module_name == "librosa":
            print(f"read audio sr: {sample_rate}")
            audio_data, _ = librosa.load( audio_path, sr=sample_rate, mono=mono)
        
        elif module_name == 'torchaudio':
            audio_data, original_samplerate = torchaudio.load(audio_path) #[channel, time], int
            if sample_rate is not None and sample_rate != original_samplerate:
                audio_data = torchaudio.transforms.Resample(orig_freq = original_samplerate, new_freq = sample_rate)(audio_data)
                
        if mono is not None:
            if mono and audio_data.shape[0] == 2:
                audio_data = np.mean(audio_data,axis=1)
            elif not mono and len(audio_data.shape) == 1:
                stereo_audio = np.zeros((2,len(audio_data)))
                stereo_audio[0,...] = audio_data
                stereo_audio[1,...] = audio_data
                audio_data = stereo_audio
        
        assert ((len(audio_data.shape)==1) or ((len(audio_data.shape)==2) and audio_data.shape[0] in [1,2])),f'read audio shape problem: {audio_data.shape}'
            
        return audio_data
    
    @staticmethod
    def stereo_to_modo(audio_data:Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        audio_data = np.mean(audio_data,axis=1)
        return audio_data
    
    @staticmethod
    def mono_to_stereo(audio_data:Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        stereo_audio = np.zeros((2,len(audio_data)))
        stereo_audio[0,...] = audio_data
        stereo_audio[1,...] = audio_data
        audio_data = stereo_audio
        return audio_data

    @staticmethod
    def normalize_audio_volume(audio_input:ndarray,sr:int, target_dBFS = -30):
        audio = UtilAudio.change_dtype(audio=audio_input,current_dtype='float64',target_dtype='int32')#UtilAudio.float64_to_int32(audio_input)
        audio_segment = AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=audio.dtype.itemsize, channels=1)
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        normalizedsound = audio_segment.apply_gain(change_in_dBFS)
        return UtilAudio.change_dtype(audio=np.array(normalizedsound.get_array_of_samples()),current_dtype='int32',target_dtype='float64') #UtilAudio.int32_to_float64(np.array(normalizedsound.get_array_of_samples()))

    @staticmethod
    def energy_unify(estimated, original, eps = 1e-12):
        target = UtilAudio.pow_norm(estimated, original) * original
        target /= UtilAudio.pow_p_norm(original) + eps
        return estimated, target

    @staticmethod
    def pow_norm(s1, s2):
        return torch.sum(s1 * s2)

    @staticmethod
    def pow_p_norm(signal):
        return torch.pow(torch.norm(signal, p=2), 2)
    
    @staticmethod
    def get_segment_index_list(audio:ndarray, #[time]
                               sample_rate:int,
                               segment_sample_length:int,
                               hop_seconds:float = 0.1
                               ) -> list:
        begin_sample:int = 0
        hop_samples = int(hop_seconds * sample_rate)
        segment_index_list = list()
        while (begin_sample == 0) or (begin_sample + segment_sample_length < len(audio)):
            segment_index_list.append({'begin':begin_sample, 'end':begin_sample + segment_sample_length})
            begin_sample += hop_samples
        return segment_index_list

