from typing import Literal

import os
import unicodedata
from tqdm import tqdm
from torch_jaekwon.Util.Util import Util 
from torch_jaekwon.Util.UtilData import UtilData
from torch_jaekwon.Util.TableMaker import HTMLUtil

TD_WIDTH = 300

class AudioListeningTableMaker:
    @staticmethod
    def get_yaml_example(output_dir:str = './') -> None:
        file_path:str = f'{os.path.dirname(__file__)}/audio_table_config_example.yaml'
        Util.system(f'cp {file_path} {output_dir}')

    @staticmethod
    def make_table_from_config_path(yaml_path:str, output_dir:str = None, max_num_tr:int = 5) -> None:
        meta_data:dict = UtilData.yaml_load(yaml_path)
        AudioListeningTableMaker.make_table(output_dir = output_dir, max_num_tr = max_num_tr, **meta_data)

    @staticmethod
    def make_table(
            title:str = '',
            sub_title:str = '',
            audio_dir_meta_list:list = list(),
            audio_name_list:list = None,
            audio_name_list_ref_dir:str = None,
            output_dir:str = None,
            sr:int = 44100,
            spec_type:Literal['mel', 'stft', 'x'] = 'mel',
            max_num_tr:int = 5,
            return_html:bool = False
        ):
        if output_dir is None: output_dir = f'./output/{title}'
        if audio_name_list is None and audio_name_list_ref_dir is not None:
            audio_name_list = [meta['file_name'] for meta in UtilData.walk(audio_name_list_ref_dir)]
        html_util = HTMLUtil(output_dir=output_dir)
        html_list = list()
        html_list.append(html_util.get_html_text(title))
        html_list.append(html_util.get_html_text(sub_title, tag='h3'))

        html_dict_list = list()
        for audio_name in tqdm(audio_name_list, desc='audio list'):
            if len(html_dict_list) > max_num_tr or audio_name is None:
                html_list += html_util.get_table_html_list(html_dict_list)
                html_dict_list = list()
            if audio_name is None: continue

            table_row_dict_audio = dict()
            table_row_dict_spec = dict()
            table_row_dict_audio['name'] = f'''<div style="width:{TD_WIDTH}px">{audio_name}<div>'''
            for audio_dir_meta in audio_dir_meta_list:
                audio_dir_name = audio_dir_meta.get('name', audio_dir_meta['dir'].split('/')[-1])
                audio_path:str = AudioListeningTableMaker.get_audio_path(audio_name, audio_dir_meta)
                if os.path.isfile(audio_path):
                    media_html_dict:dict = html_util.get_html_audio(audio_path = audio_path, sample_rate=sr, spec_type=spec_type)
                else:
                    file_strict:bool = audio_dir_meta.get('file_strict', True)
                    if file_strict: 
                        Util.print(f'File not found: {audio_path}', msg_type='error')
                        raise FileNotFoundError
                    else:
                        media_html_dict = {'audio': f'<div style="width:{TD_WIDTH}px"> X <div>', 'spec': f'<div style="width:{TD_WIDTH}px"> X <div>'}

                table_row_dict_audio[audio_dir_name] = media_html_dict['audio']
                table_row_dict_spec[audio_dir_name] = media_html_dict['spec']
            html_dict_list.append(table_row_dict_audio)
            html_dict_list.append(table_row_dict_spec)
        
        if len(html_dict_list) > 0: html_list += html_util.get_table_html_list(html_dict_list)
        if return_html: return html_list
        else: html_util.save_html(html_list)
    
    @staticmethod
    def get_audio_path(audio_name:str, audio_dir_meta:dict) -> str:
        ext = 'wav'
        if audio_dir_meta.get('use_only_name', True): audio_name = audio_name.split('/')[-1]
        audio_name_pre_post_fix = audio_dir_meta.get('audio_name_pre_post_fix',['',''])
        audio_file_name:str = audio_dir_meta.get('audio_name', None)
        if audio_file_name is None:
            audio_file_name = f'{audio_name_pre_post_fix[0]}{audio_name}{audio_name_pre_post_fix[1]}.{ext}'
        else:
            audio_file_name = f'{audio_name}/{audio_name_pre_post_fix[0]}{audio_file_name}{audio_name_pre_post_fix[1]}.{ext}'
        audio_path:str = f"{audio_dir_meta['dir']}/{audio_file_name}"
        
        if os.path.isfile(audio_path): return audio_path
        else: return unicodedata.normalize("NFC", audio_path)
