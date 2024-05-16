from typing import Optional, Literal, Union, Final, List
from numpy import ndarray

from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import resample_poly

try: import torch 
except: print('import error: torch')
try: from torch import Tensor
except: print('')
try: import torchaudio
except: print('import error: torch')
try: from pydub import AudioSegment  
except: print('import error: pydub')

from TorchJaekwon.Util.UtilData import UtilData

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
    def resample_audio(audio:Union[ndarray, Tensor], #[shape=(channel, num_samples) or (num_samples)]
                       origin_sr:int,
                       target_sr:int,
                       resample_module:Literal['librosa', 'resample_poly', 'torchaudio'] = 'librosa',
                       resample_type:str = "kaiser_fast"):
        if(origin_sr == target_sr): return audio
        print(f"resample audio {origin_sr} to {target_sr}")
        if resample_module == 'librosa':
            return librosa.resample(audio, orig_sr=origin_sr, target_sr=target_sr, res_type=resample_type)
        elif resample_module == 'resample_poly':
            return resample_poly(x = audio, up = target_sr, down = origin_sr)
        elif resample_module == 'torchaudio':
            return torchaudio.transforms.Resample(orig_freq = origin_sr, new_freq = target_sr)(audio)
    
    @staticmethod
    def read(audio_path:str,
             sample_rate:Optional[int] = None,
             mono:Optional[bool] = None,
             module_name:Literal['soundfile','librosa', 'torchaudio'] = 'soundfile',
             return_type:Union[ndarray, Tensor] = ndarray
            ) -> Union[ndarray, Tensor]: #[shape=(channel, num_samples) or (num_samples)]
        
        if module_name == "soundfile":
            audio_data, original_samplerate = sf.read(audio_path)
            if len(audio_data.shape) > 1 : audio_data = audio_data.T

            if sample_rate is not None and sample_rate != original_samplerate:
                print(f"resample audio {original_samplerate} to {sample_rate}")
                audio_data = UtilAudio.resample_audio(audio_data,original_samplerate,sample_rate)

        elif module_name == "librosa":
            print(f"read audio sr: {sample_rate}")
            audio_data, original_samplerate = librosa.load( audio_path, sr=sample_rate, mono=mono)
        
        elif module_name == 'torchaudio':
            audio_data, original_samplerate = torchaudio.load(audio_path) #[channel, time], int
            if sample_rate is not None and sample_rate != original_samplerate:
                audio_data = torchaudio.transforms.Resample(orig_freq = original_samplerate, new_freq = sample_rate)(audio_data)
                
        if mono is not None:
            if mono and len(audio_data.shape) == 2 and audio_data.shape[0] == 2:
                audio_data = torch.mean(audio_data,axis=0)  if isinstance(audio_data, torch.Tensor) else np.mean(audio_data,axis=0) 
            elif not mono and len(audio_data.shape) == 1:
                stereo_audio = np.zeros((2,len(audio_data)))
                stereo_audio[0,...] = audio_data
                stereo_audio[1,...] = audio_data
                audio_data = stereo_audio
        
        assert ((len(audio_data.shape)==1) or ((len(audio_data.shape)==2) and audio_data.shape[0] in [1,2])),f'[read audio shape problem] path: {audio_path} shape: {audio_data.shape}'
            
        return audio_data, original_samplerate if sample_rate is None else sample_rate
    
    @staticmethod
    def stereo_to_mono(audio_data:Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
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
    def normalize_volume(audio_input:ndarray,sr:int, target_dBFS = -30):
        audio = UtilAudio.change_dtype(audio=audio_input,current_dtype='float64',target_dtype='int32')#UtilAudio.float64_to_int32(audio_input)
        audio_segment = AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=audio.dtype.itemsize, channels=1)
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        normalizedsound = audio_segment.apply_gain(change_in_dBFS)
        return UtilAudio.change_dtype(audio=np.array(normalizedsound.get_array_of_samples()),current_dtype='int32',target_dtype='float64') #UtilAudio.int32_to_float64(np.array(normalizedsound.get_array_of_samples()))
    
    @staticmethod
    def normalize_by_fro_norm(audio_input:Tensor #[batch, channel, time]
                              ) -> Tensor:
        original_shape:tuple = audio_input.shape
        audio = audio_input.reshape(original_shape[0], -1)
        audio = audio/torch.norm(audio, p="fro", dim=1, keepdim=True)
        audio = audio.reshape(*original_shape)
        return audio

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
    
    @staticmethod
    def merge_audio_w_cross_fade(audio_list:List[ndarray],
                                 segment_length:int,
                                 overlap_length:int
                                 ) -> ndarray:
        '''
        reference from https://github.com/nkandpa2/music_enhancement/blob/master/scripts/generate_from_wav.py
        '''
        output_audio:ndarray = np.zeros(len(audio_list) * segment_length - (len(audio_list) - 1) * overlap_length)
        hop_length:int = segment_length - overlap_length
        cross_fade_in:ndarray = np.linspace(0, 1, overlap_length)
        cross_fade_out:ndarray = 1 - cross_fade_in
        
        for i in range(0,len(audio_list)):
            start_idx:int = i * hop_length
            audio_list[i][:overlap_length] *= cross_fade_in
            audio_list[i][-overlap_length:] *= cross_fade_out
            output_audio[start_idx:start_idx+segment_length] += audio_list[i]
        return output_audio
    
    @staticmethod
    def analyze_audio_dataset(data_dir:str, 
                              result_save_dir:str,
                              sanity_check_sr:Union[int,List[int]] = None,
                              save_each_meta:bool = False
                              ) -> None:
        total_meta_dict:dict = {
            'total_sample_length': 0,
            'total_duration_second': 0,
            'total_duration_minutes': 0,
            'total_duration_hours': 0,

            'longest_sample_meta': {
                'file_name': '',
                'sample_length':0
            }
        }
        if sanity_check_sr is not None: total_meta_dict['sample_rate'] = sanity_check_sr
        
        audio_meta_data_list = UtilData.walk(dir_name=data_dir, ext=['.wav', '.mp3', '.flac'])
        for meta_data in tqdm(audio_meta_data_list):
            audio, sr = UtilAudio.read(meta_data['file_path'], mono=True)
            if sanity_check_sr is not None: 
                if isinstance(sanity_check_sr, int): assert sr == sanity_check_sr, f'''{meta_data['file_path']}'s sample rate is {sr}'''
                if isinstance(sanity_check_sr, list): assert sr in sanity_check_sr, f'''{meta_data['file_path']}'s sample rate is {sr}'''
            meta_data_of_this_file = {
                'file_name': meta_data['file_name'],
                'sample_length': len(audio),
                'duration_second': len(audio) / sr
            }
            save_dir:str = meta_data['dir_path'].replace(data_dir, result_save_dir)
            if save_each_meta: UtilData.pickle_save(f'''{save_dir}/{meta_data['file_name']}.pkl''', meta_data_of_this_file)

            total_meta_dict['total_sample_length'] += meta_data_of_this_file['sample_length']
            if total_meta_dict['longest_sample_meta']['sample_length'] < meta_data_of_this_file['sample_length']:
                total_meta_dict['longest_sample_meta'] = meta_data_of_this_file
        
        total_meta_dict['total_duration_second'] = total_meta_dict['total_sample_length'] / sr
        total_meta_dict['total_duration_minutes'] = total_meta_dict['total_duration_second'] / 60
        total_meta_dict['total_duration_hours'] = total_meta_dict['total_duration_second'] / 3600
        UtilData.yaml_save(save_path = f'{result_save_dir}/meta.yaml', data = total_meta_dict)
        

