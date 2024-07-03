from typing import Literal, Tuple
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip

class UtilVideo:
    @staticmethod
    def attach_audio_to_video(video_path:str, 
                              audio_path:str, 
                              output_path:str, 
                              fps:int=30, 
                              video_duration_sec:float = None
                              ) -> VideoFileClip:
        video_clip = VideoFileClip(video_path).set_fps(fps)
        if video_duration_sec is not None:
            video_clip = video_clip.subclip(0, video_duration_sec)
        video_clip = video_clip.set_audio(AudioFileClip(audio_path))
        video_clip.write_videofile(output_path, audio=True, fps=fps, verbose=False, logger=None)
        return video_clip
    
    @staticmethod
    def attach_audio_to_img(image_path:str,
                            audio_path:str,
                            output_path:str = 'output.mp4',
                            video_size:Tuple[int,int]=(1920,1080),
                            module:Literal['moviepy', 'ffmpeg'] = 'moviepy'
                            ):
        if module == 'moviepy':
            audio = AudioFileClip(audio_path)
            image_clip = ImageClip(image_path).set_duration(audio.duration).resize(newsize=video_size)
            video = image_clip.set_audio(audio)
            video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
        elif module == 'ffmpeg':
            subprocess.run([
                'ffmpeg', '-loop', '1', '-i', image_path, '-i', audio_path,
                '-vf', f'scale={video_size[0]}:{video_size[1]}', '-c:v', 'libx264', 
                '-c:a', 'aac', '-shortest', output_path
            ])
