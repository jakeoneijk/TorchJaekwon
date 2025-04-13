from typing import Optional, Literal, Union, Final, List
from numpy import ndarray

import os
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

from .util_data import UtilData

DATA_TYPE_MIN_MAX_DICT:Final[dict] = {'float32':(-1,1), 'float64':(-1,1), 'int16':(-2**15, 2**15-1), 'int32':(-2**31,2**31-1)}

class UtilAudio:
    @staticmethod
    def change_dtype(
        audio:ndarray,
        current_dtype:Literal['float32', 'float64', 'int16', 'int32'],
        target_dtype:Literal['float32', 'float64', 'int16', 'int32']
    ) -> ndarray:
        audio = np.clip(audio, a_min = DATA_TYPE_MIN_MAX_DICT[current_dtype][0], a_max = DATA_TYPE_MIN_MAX_DICT[current_dtype][1])
        audio = audio / DATA_TYPE_MIN_MAX_DICT[current_dtype][1]
        audio = (audio * DATA_TYPE_MIN_MAX_DICT[target_dtype][1])
        audio = audio.astype(getattr(np,target_dtype))
        return audio
    
    @staticmethod
    def resample_audio(
        audio:Union[ndarray, Tensor], #[shape=(channel, num_samples) or (num_samples)]
        origin_sr:int,
        target_sr:int,
        resample_module:Literal['librosa', 'resample_poly', 'torchaudio'] = 'torchaudio',
        resample_type:str = "kaiser_fast",
    ) -> Union[ndarray, Tensor]:
        if(origin_sr == target_sr): return audio
        #print(f"resample audio {origin_sr} to {target_sr}")
        if resample_module == 'librosa':
            return librosa.resample(audio, orig_sr=origin_sr, target_sr=target_sr, res_type=resample_type)
        elif resample_module == 'resample_poly':
            return resample_poly(x = audio, up = target_sr, down = origin_sr)
        elif resample_module == 'torchaudio':
            if isinstance(audio, ndarray): audio = torch.FloatTensor(audio)
            #transforms.Resample precomputes and caches the kernel used for resampling, while functional.resample computes it on the fly
            #so using torchaudio.transforms.Resample will result in a speedup when resampling multiple waveforms using the same parameters
            resampler = torchaudio.transforms.Resample(orig_freq = origin_sr, new_freq = target_sr).to(audio.device)
            return resampler(audio)
    
    @staticmethod
    def read(
        audio_path:str,
        # result parameters
        sample_rate:Optional[int] = None, # Output sample rate. If None, original sample rate will be used
        mono:Optional[bool] = None,
        sample_length:Optional[int] = None,
        return_type:Union[ndarray, Tensor] = Tensor,
        # segment parameters
        start:Optional[int] = None,
        end:Optional[int] = None,
        segment_type:Literal['time','sample'] = 'sample',
        origin_sample_rate:Optional[int] = None,
        # module parameters
        module_name:Literal['soundfile','librosa', 'torchaudio'] = 'torchaudio',
    ) -> Union[ndarray, Tensor]: # shape=(channel, num_samples)
        # error check
        if segment_type == 'time': assert origin_sample_rate is not None, f'[Error] origin_sample_rate must be given when segment_type is time'
        if start is not None: assert module_name == 'torchaudio', f'[Error] currently only torchaudio module supports start and end parameter'
        
        if module_name == "soundfile":
            audio_data, original_sr = sf.read(audio_path)
            if len(audio_data.shape) > 1 : audio_data = audio_data.T
            if sample_rate is not None and sample_rate != original_sr:
                audio_data = UtilAudio.resample_audio(audio_data,original_sr,sample_rate)
        elif module_name == "librosa":
            audio_data, original_sr = librosa.load( audio_path, sr=sample_rate, mono=mono)
        elif module_name == 'torchaudio':
            if start is None:
                frame_offset = 0
            else:
                frame_offset = round(start * origin_sample_rate) if segment_type == 'time' else start
            if end is None:
                num_frames = -1
            else:
                num_frames = round((end - start) * origin_sample_rate) if segment_type == 'time' else end - start
            audio_data, original_sr = torchaudio.load(
                audio_path, 
                frame_offset = frame_offset, 
                num_frames = num_frames
            )
            if origin_sample_rate is not None: assert origin_sample_rate == original_sr, f'[Error] origin_sample_rate is not same with original sample rate'
            if sample_rate is not None and sample_rate != original_sr:
                audio_data = UtilAudio.resample_audio(audio = audio_data, origin_sr=original_sr, target_sr = sample_rate, resample_module='torchaudio')
        
        if sample_length is not None:
            audio_data = UtilData.fix_length(audio_data, sample_length, dim = -1)
            
        if mono is not None:
            if mono and len(audio_data.shape) == 2 and audio_data.shape[0] == 2:
                audio_data = torch.mean(audio_data, axis=0, keepdim=True)  if isinstance(audio_data, torch.Tensor) else np.mean(audio_data,axis=0) 
            elif not mono and (len(audio_data.shape) == 1 or audio_data.shape[0] == 1):
                stereo_audio = torch.zeros((2,len(audio_data.squeeze())))
                stereo_audio[0,...] = audio_data.squeeze()
                stereo_audio[1,...] = audio_data.squeeze()
                audio_data = stereo_audio
        
        if isinstance(audio_data, Tensor) and return_type == ndarray:
            audio_data = audio_data.cpu().detach().numpy()

        assert (((len(audio_data.shape)==2) and audio_data.shape[0] in [1,2])),f'[read audio shape problem] path: {audio_path} shape: {audio_data.shape}'
            
        return audio_data, original_sr if sample_rate is None else sample_rate
    
    @staticmethod
    def write(
        audio_path:str,
        audio:Union[ndarray, Tensor],
        sample_rate:int,
    ) -> None:
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        if isinstance(audio, Tensor):
            audio = audio.squeeze().cpu().detach().numpy()
        assert len(audio.shape) <= 2, f'[Error] shape of {audio_path}: {audio.shape}'
        if len(audio.shape) == 2 and audio.shape[0] < audio.shape[1]: audio = audio.T
        sf.write(file = audio_path, data = audio, samplerate = sample_rate)
    
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
    def normalize_by_fro_norm(
        audio_input:Tensor #[batch, channel, time]
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
    def get_segment_index_list(
        audio:ndarray, #[time]
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
    def audio_to_batch(
        audio:Tensor, #[Length]
        segment_length:int,
        overlap_length:int = 48000 #recommend: int(sr * 0.5)
    ) -> Tensor:
        
        assert len(audio.shape) == 1, f'[Error] audio shape must be 1, but {audio.shape}'
        start_idx:int = 0
        audio_list = list()
        while start_idx < len(audio):
            audio_segment = audio[start_idx:start_idx+segment_length]
            audio_segment = UtilData.fix_length(audio_segment, segment_length)
            audio_list.append(audio_segment)
            start_idx += segment_length - overlap_length
        return torch.stack(audio_list)
    
    @staticmethod
    def merge_batch_w_cross_fade(
        batch_audio:Union[List[ndarray],ndarray,Tensor],
        segment_length:int,
        overlap_length:int = 48000 #recommend: int(sr * 0.5)
    ) -> ndarray:
        '''
        reference from https://github.com/nkandpa2/music_enhancement/blob/master/scripts/generate_from_wav.py
        '''
        if isinstance(batch_audio, ndarray) and len(batch_audio.shape) == 1:
            batch_audio = [batch_audio]
        output_audio_length:int = len(batch_audio) * segment_length - (len(batch_audio) - 1) * overlap_length
        output_audio:Union[ndarray,Tensor] = torch.zeros(output_audio_length) if isinstance(batch_audio, torch.Tensor) else np.zeros(output_audio_length)
        hop_length:int = segment_length - overlap_length
        
        cross_fade_in:ndarray = np.linspace(0, 1, overlap_length)
        cross_fade_out:ndarray = 1 - cross_fade_in
        if isinstance(batch_audio, torch.Tensor):
            cross_fade_in = torch.tensor(cross_fade_in, device = batch_audio.device)
            cross_fade_out = torch.tensor(cross_fade_out, device = batch_audio.device)
        
        for i in range(0,len(batch_audio)):
            start_idx:int = i * hop_length
            if i != 0:
                batch_audio[i][:overlap_length] *= cross_fade_in
            if i != len(batch_audio) - 1:
                batch_audio[i][-overlap_length:] *= cross_fade_out
            output_audio[start_idx:start_idx+segment_length] += batch_audio[i]
        return output_audio
    
    @staticmethod
    def analyze_audio_dataset(
        data_dir_list:Union[str, list], 
        result_save_dir:str = './meta',
        sanity_check_sr:Union[int,List[int]] = None,
        save_each_meta:bool = False
    ) -> None:
        if isinstance(data_dir_list, str): data_dir_list = [data_dir_list]
        result_meta_dict = dict()
        
        for data_dir in tqdm(data_dir_list, desc="Dataset list"):
            dir_name:str = UtilData.get_file_name(data_dir)
            result_meta_dict[dir_name]:dict = {
                'total_duration_second': 0,
                'total_duration_minutes': 0,
                'total_duration_hours': 0,

                'longest_sample_meta': {
                    'file_name': '',
                    'duration_second':0
                },

                'sample_rate': list(),
                'error_file_list': list()
            }
            if sanity_check_sr is not None: result_meta_dict[dir_name]['sample_rate'] = sanity_check_sr
            
            audio_meta_data_list = UtilData.walk(dir_name=data_dir, ext=['.wav', '.mp3', '.flac'])
            for meta_data in tqdm(audio_meta_data_list):
                try:
                    audio, sr = UtilAudio.read(meta_data['file_path'], mono=True)
                except:
                    print(f'Error: {meta_data["file_path"]}')
                    result_meta_dict[dir_name]['error_file_list'].append(meta_data['file_path'])
                    continue
                if sanity_check_sr is not None: 
                    if isinstance(sanity_check_sr, int): assert sr == sanity_check_sr, f'''{meta_data['file_path']}'s sample rate is {sr}'''
                    if isinstance(sanity_check_sr, list): assert sr in sanity_check_sr, f'''{meta_data['file_path']}'s sample rate is {sr}'''
                
                meta_data_of_this_file = {
                    'file_name': meta_data['file_name'],
                    'file_path': os.path.abspath(meta_data['file_path']),
                    'sample_length': audio.shape[-1],
                    'sample_rate': sr,
                }
                meta_data_of_this_file['duration_second'] = meta_data_of_this_file['sample_length'] / meta_data_of_this_file['sample_rate']
                
                if save_each_meta: 
                    UtilData.pickle_save(f"{result_save_dir}/per_sample/{dir_name}/{meta_data['file_path'].split(dir_name)[-1]}.pkl".replace('//','/'), meta_data_of_this_file)

                result_meta_dict[dir_name]['total_duration_second'] += meta_data_of_this_file['duration_second']
                if result_meta_dict[dir_name]['longest_sample_meta']['duration_second'] < meta_data_of_this_file['duration_second']:
                    result_meta_dict[dir_name]['longest_sample_meta'] = meta_data_of_this_file
                if meta_data_of_this_file['sample_rate'] not in result_meta_dict[dir_name]['sample_rate']:
                    result_meta_dict[dir_name]['sample_rate'].append(meta_data_of_this_file['sample_rate'])

            result_meta_dict[dir_name]['total_duration_minutes'] = result_meta_dict[dir_name]['total_duration_second'] / 60
            result_meta_dict[dir_name]['total_duration_hours'] = result_meta_dict[dir_name]['total_duration_second'] / 3600
        UtilData.yaml_save(save_path = f'{result_save_dir}/meta.yaml', data = result_meta_dict)
    
    @staticmethod
    def resample_audio_dataset(
        data_dir_list:Union[str, list], 
        sr:int,
        save_dir:str = None,
    ) -> None:
        if isinstance(data_dir_list, str): data_dir_list = [data_dir_list]
        for data_dir in tqdm(data_dir_list, desc="Dataset list"):
            audio_meta_data_list = UtilData.walk(dir_name=data_dir, ext=['.wav', '.mp3', '.flac'])
            for meta_data in tqdm(audio_meta_data_list):
                audio, _ = librosa.load(meta_data['file_path'], sr = sr)
                if save_dir == None:
                    audio_dir_new = meta_data['dir_path'].replace(meta_data['dir_name'], f"{meta_data['dir_name']}_{sr}")
                UtilAudio.write(f"{audio_dir_new}/{meta_data['file_name']}.wav", audio, sr)
        
        

