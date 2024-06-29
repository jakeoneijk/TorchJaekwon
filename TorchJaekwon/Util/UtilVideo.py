from moviepy.editor import VideoFileClip, AudioFileClip

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