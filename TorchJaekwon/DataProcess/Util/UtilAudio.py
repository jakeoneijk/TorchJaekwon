from numpy import ndarray
import numpy as np
import soundfile as sf
import librosa

try:
    import torch
    from pydub import AudioSegment, effects  
except:
    print('import error: pydub')

class UtilAudio:
    @staticmethod
    def float32_to_int16(x: np.float32) -> np.int16:
        #-2**15 to 2**15-1
        x = np.clip(x, a_min=-1, a_max=1)
        return (x * 32767.0).astype(np.int16)
    
    @staticmethod
    def int16_to_float32(x: np.int16) -> np.float32:
        return (x / 32767.0).astype(np.float32)
    
    @staticmethod
    def int32_to_float64(x: np.int32) -> np.float64:
        return (x / (2**31 - 1)).astype(np.float64)
    
    @staticmethod
    def float64_to_int32(x: np.float64) -> np.int32:
        return (x * (2**31 - 1)).astype(np.int32)
    
    @staticmethod
    def resample_audio(audio,origin_sr,target_sr,resample_type = "kaiser_fast"):
        if(origin_sr == target_sr): return audio
        print(f"resample audio {origin_sr} to {target_sr}")
        return librosa.core.resample(audio, orig_sr=origin_sr, target_sr=target_sr, res_type=resample_type)
    
    @staticmethod
    def read_audio_fix_channels_sr(
            audio_path:str,sample_rate=None, mono=False,read_type=["soundfile","librosa",][0]
            ) -> ndarray: #[shape=(channel, num_samples) or (num_samples)]
        if read_type == "soundfile":
            audio_data, original_samplerate = sf.read(audio_path)
            audio_data = audio_data.T

            if sample_rate is not None and sample_rate != original_samplerate:
                print(f"resample audio {original_samplerate} to {sample_rate}")
                audio_data = UtilAudio.resample_audio(audio_data,original_samplerate,sample_rate)

        elif read_type == "librosa":
            print(f"read audio sr: {sample_rate}")
            audio_data, _ = librosa.load( audio_path, sr=sample_rate, mono=mono)
        
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
    def normalize_audio_volume(audio_input:ndarray,sr:int, target_dBFS = -30):
        audio = UtilAudio.float64_to_int32(audio_input)
        audio_segment = AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=audio.dtype.itemsize, channels=1)
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        normalizedsound = audio_segment.apply_gain(change_in_dBFS)
        return UtilAudio.int32_to_float64(np.array(normalizedsound.get_array_of_samples()))

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

