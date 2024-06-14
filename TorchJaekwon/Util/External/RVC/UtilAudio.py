import numpy as np

class UtilAudio:
    @staticmethod
    def norm_audio(audio: np.ndarray, #[time]
                   max: float = 0.9,
                   alpha: float = 0.75,
                   ) -> np.ndarray:
        tmp_max = np.abs(audio).max()
        assert tmp_max <= 2.5, "The maximum value of the audio is too high."
        
        tmp_audio = (tmp_audio / tmp_max * (max * alpha)) + (
            1 - alpha
        ) * tmp_audio
        return tmp_audio.astype(np.float32)