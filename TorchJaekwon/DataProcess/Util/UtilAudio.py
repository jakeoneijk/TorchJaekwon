from numpy import ndarray
import numpy as np
import soundfile as sf
import librosa

class UtilAudio:
    def float32_to_int16(self, x: np.float32) -> np.int16:
        x = np.clip(x, a_min=-1, a_max=1)
        return (x * 32767.0).astype(np.int16)
    
    def int16_to_float32(self, x: np.int16) -> np.float32:
        return (x / 32767.0).astype(np.float32)
    
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
