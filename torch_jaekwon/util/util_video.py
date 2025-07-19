from typing import Literal, Tuple, Optional, Union

import os
import subprocess
import numpy as np
try: from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip
except: print('moviepy is not installed. Please install it using `pip install moviepy`')
try: from pytube import YouTube
except: print('pytube is not installed. Please install it using `pip install pytube`')
try: from pytubefix import YouTube as YouTubeFix
except: print('pytubefix is not installed. Please install it using `pip install pytubefix`')
try: from pydub import AudioSegment
except: print('pydub is not installed. Please install it using `pip install pydub`')

from torch_jaekwon.util import util

class UtilVideo:
    @staticmethod
    def read(
        file_path:str,
        fps:Optional[int] = None,
        start_sec:Optional[int] = None,
        end_sec:Optional[int] = None,
    ) -> VideoFileClip:
        assert [start_sec is None, end_sec is None].count(None) != 1, "Either both start_sec and end_sec should be None or both should be provided."
        video = VideoFileClip(file_path)
        if fps is not None:
            video = video.set_fps(fps)
        if start_sec is not None:
            video = video.subclip(start_sec, end_sec)
        return video
    
    @staticmethod
    def write(
        file_path:str,
        video:VideoFileClip,
        fps:int = None,
        codec='libx264',
        audio_codec:Literal['aac', 'pcm_s16le', 'pcm_s32le'] = None,
        logger=None
    ) -> None:
        util.make_parent_dir(file_path)
        video.write_videofile(
            filename = file_path,
            fps = fps, 
            codec=codec,
            audio = True,
            audio_codec = audio_codec,
            logger=logger
        )
    
    @staticmethod
    def download_youtube(
        url:str,
        save_path:str,
        resolution:Literal['360p', '480p', '720p', '1080p'] = '1080p',
        audio:bool = False,
    ) -> VideoFileClip:
        util.make_parent_dir(save_path)
        def download(yt, save_path, resolution) -> str:
            resolution_list:list = ['360p', '480p', '720p', '1080p']
            resolution_idx:int = resolution_list.index(resolution)
            yt_args = {'file_extension': 'mp4'}
            if audio:
                yt_args['progressive'] = True
            else:
                yt_args['adaptive'] = True
            steam_list = yt.streams.filter(**yt_args).order_by('resolution').desc()
            if len(steam_list) == 0:
                return None
            stream = None
            while stream is None and resolution_idx >= 0:
                stream_list_filtered = [s for s in steam_list if s.resolution == resolution_list[resolution_idx]]
                if len(stream_list_filtered) == 0: 
                    resolution_idx -= 1
                    continue
                stream = stream_list_filtered[0]
                download_path = stream.download(output_path='/'.join(save_path.split('/')[:-1]), filename=save_path.split('/')[-1])
                return download_path
            raise ValueError(f"There is no available streams.")
        try:
            return download(YouTube(url), save_path, resolution)
        except Exception as e:
            return download(YouTubeFix(url), save_path, resolution)

        
    @staticmethod
    def extract_audio_from_video(
        video_path:str,
        output_path:str = './tmp/tmp.wav',
    ) -> str:
        video = VideoFileClip(video_path)
        audio = video.audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio.write_audiofile(output_path, codec='pcm_s16le')
        return output_path

    @staticmethod
    def attach_audio_to_video(
        video_path:str, 
        audio_path:str, 
        output_path:str, 
        fps:int=30, 
        video_duration_sec:float = None,
        audio_codec:Literal['aac', 'pcm_s16le', 'pcm_s32le'] = 'aac',
    ) -> VideoFileClip:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_clip = UtilVideo.read(file_path=video_path, fps=fps)
        if video_duration_sec is not None:
            video_clip = video_clip.subclip(0, video_duration_sec)
        video_clip = video_clip.set_audio(AudioFileClip(audio_path))
        video_clip.write_videofile(
            output_path, 
            audio=True, 
            audio_codec = audio_codec,
            fps=fps, 
            verbose=False, 
            logger=None
        )
        return video_clip
    
    @staticmethod
    def attach_audio_to_img(
        image_path:str,
        audio_path:str,
        output_path:str = 'output.mkv',
        audio_codec:Literal['aac', 'pcm_s16le', 'pcm_s32le'] = 'pcm_s32le',
        audio_fps:int=44100,
        video_size:Tuple[int,int]=(1920,1080),
        module:Literal['moviepy', 'ffmpeg'] = 'moviepy'
    ) -> None:
        if module == 'moviepy':
            import PIL
            PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
            audio = AudioFileClip(audio_path)
            image_clip:ImageClip = ImageClip(image_path).set_duration(audio.duration).resize(newsize=video_size)
            video = image_clip.set_audio(audio)
            video.write_videofile(output_path, 
                                  codec='libx264', 
                                  audio_fps = audio_fps,
                                  audio_codec=audio_codec, 
                                  fps=24)
        elif module == 'ffmpeg':
            subprocess.run([
                'ffmpeg', '-loop', '1', '-i', image_path, '-i', audio_path,
                '-vf', f'scale={video_size[0]}:{video_size[1]}', '-c:v', 'libx264', 
                '-c:a', 'aac', '-shortest', output_path
            ])
