from typing import Literal, Tuple, Optional, Union

import os
import subprocess
from torio.io import StreamingMediaDecoder
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip
try: from pytube import YouTube
except: print('pytube is not installed. Please install it using `pip install pytube`')
try: from pytubefix import YouTube as YouTubeFix
except: print('pytubefix is not installed. Please install it using `pip install pytubefix`')

from torch_jaekwon.util import util, util_audio


def read_stream(
    file_path: str,
    video_stream_args_dict: dict = {'video': {'frame_rate': 24}},
    audio_stream_args_dict: dict = {'audio': {'mono': None, 'sample_rate': None}},
    duration_sec: Optional[float] = None,
    normalize_audio: bool = True,
) -> None:
    assert all([any([param_name in stream_config for param_name in ['frame_rate', 'frames_per_chunk']]) for stream_config in video_stream_args_dict.values()]), "Each video stream config must contain either 'frame_rate' or 'frames_per_chunk'."
    default_video_stream_config = {'format': 'rgb24'}

    reader = StreamingMediaDecoder(file_path)

    video_stream_key_list:list = list(video_stream_args_dict.keys())
    for stream_key in video_stream_key_list:
        video_stream_args = video_stream_args_dict[stream_key]
        if 'frames_per_chunk' not in video_stream_args:
            video_stream_args['frames_per_chunk'] = int(video_stream_args['frame_rate'] * duration_sec)
        video_stream_args = {**default_video_stream_config, **video_stream_args}
        reader.add_basic_video_stream(**video_stream_args)

    audio_stream_key_list:list = list(audio_stream_args_dict.keys())
    for stream_key in audio_stream_key_list:
        reader.add_basic_audio_stream(frames_per_chunk=2**30)
    
    reader.fill_buffer()
    data_chunk_list = reader.pop_chunks()
    data_dict = dict()
    for i, stream_key in enumerate(video_stream_key_list):
        video = data_chunk_list[i]
        assert video is not None, util.log(f"Can't read stream of {file_path}", msg_type='error')
        assert video.shape[0] >= video_stream_args_dict[stream_key]['frames_per_chunk'], util.log(f"Stream {stream_key} of {file_path} has less frames than expected. Expected: {video_stream_args_dict[stream_key]['frames_per_chunk']}, Actual: {video.shape[0]}", msg_type='error')
        video = video[:video_stream_args_dict[stream_key]['frames_per_chunk']]
        data_dict[stream_key] = [video, video_stream_args_dict[stream_key].get('frame_rate')]
    
    for i, stream_key in enumerate(audio_stream_key_list):
        audio = data_chunk_list[i + len(video_stream_key_list)]
        audio = util_audio.convert_audio_channels(audio.transpose(0, 1), mono=audio_stream_args_dict[stream_key].get('mono', None))
        if normalize_audio:
            assert audio.abs().max() >= 1e-6, "Audio normalization failed. The maximum absolute value of the audio is too low."
            audio = audio / audio.abs().max() * 0.95
        sample_rate:int = int(reader.get_out_stream_info(i + len(video_stream_key_list)).sample_rate)
        if duration_sec is not None:
            audio = audio[..., :int(sample_rate * duration_sec)]
        target_sample_rate:int = audio_stream_args_dict[stream_key].get('sample_rate', sample_rate)
        audio = util_audio.resample(
            audio = audio, 
            origin_sr = sample_rate,
            target_sr = target_sample_rate,
            resample_type = 'kaiser_best'
        )
        data_dict[stream_key] = [audio, target_sample_rate]
    return data_dict


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
        end_sec = min(end_sec, video.duration)
        video = video.subclip(start_sec, end_sec)
    return video


def write(
    file_path:str,
    video:VideoFileClip,
    fps:int = None,
    codec='libx264',
    audio_codec:Literal['aac', 'pcm_s16le', 'pcm_s32le'] = 'aac',
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


def download_youtube(
    url:str = None,
    video_id:str = None,
    save_path:str = None,
    resolution:Literal['360p', '480p', '720p', '1080p'] = '1080p',
    audio:bool = False,
) -> VideoFileClip:
    assert url is not None or video_id is not None, "Either url or video_id should be provided."
    if video_id is not None:
        url = f'https://www.youtube.com/watch?v={video_id}'
    util.make_parent_dir(save_path)
    def download(yt, save_path, resolution) -> str:
        yt_args = {'file_extension': 'mp4'}
        if audio:
            yt_args['progressive'] = True
        else:
            yt_args['adaptive'] = True
        try:
            stream_list = yt.streams.filter(**yt_args).order_by('resolution').desc()
            stream_list = [stream for stream in stream_list if util.get_num_in_str(stream.resolution)[0] <=  util.get_num_in_str(resolution)[0]]
            stream = stream_list[0]
            if util.get_num_in_str(stream.resolution)[0] != util.get_num_in_str(resolution)[0]:
                util.log(f"Warning: The requested resolution {resolution} is not available. Downloading the closest available resolution: {stream.resolution}", msg_type='warning')
            download_path = stream.download(output_path='/'.join(save_path.split('/')[:-1]), filename=save_path.split('/')[-1])
            return download_path
        except Exception as e:
            if os.path.exists(save_path): os.remove(save_path)
            util.log(f"Download error: {e}", msg_type='error')
            return None
    try:
        return download(YouTube(url), save_path, resolution)
    except Exception as e:
        return download(YouTubeFix(url), save_path, resolution)

    

def extract_audio_from_video(
    video_path:str,
    output_path:str = './tmp/tmp.wav',
) -> str:
    video = VideoFileClip(video_path)
    audio = video.audio
    util.make_parent_dir(output_path)
    audio.write_audiofile(output_path, codec='pcm_s16le')
    return output_path


def attach_audio_to_video(
    video_path:str, 
    audio_path:str, 
    output_path:str, 
    fps:int=30, 
    video_duration_sec:float = None,
    audio_codec:Literal['aac', 'pcm_s16le', 'pcm_s32le'] = 'aac',
) -> VideoFileClip:
    util.make_parent_dir(output_path)
    video_read_args = {'file_path': video_path, 'fps': fps}
    if video_duration_sec is not None:
        video_read_args['start_sec'] = 0
        video_read_args['end_sec'] = video_duration_sec
    video_clip = UtilVideo.read(**video_read_args)
    video_clip = video_clip.set_audio(AudioFileClip(audio_path))
    UtilVideo.write(
        file_path = output_path, 
        video = video_clip, 
        fps = fps,
        audio_codec = audio_codec,
    )
    return video_clip


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
