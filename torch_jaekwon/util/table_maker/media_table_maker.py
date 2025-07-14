from typing import Literal

import os
import unicodedata
from tqdm import tqdm
from .. import Util, UtilData
from . import HTMLUtil

TD_WIDTH = 300
BLANK_COMPONENT = f'<div style="width:{TD_WIDTH}px"> X <div>'

class MediaTableMaker:
    @staticmethod
    def get_yaml_example(output_dir:str = './') -> None:
        file_path:str = f'{os.path.dirname(__file__)}/media_table_config_example.yaml'
        Util.system(f'cp {file_path} {output_dir}')

    @staticmethod
    def make_table_from_config_path(yaml_path:str, output_dir:str = None, max_num_tr:int = 5) -> None:
        meta_data:dict = UtilData.yaml_load(yaml_path)
        if meta_data['title'] is None: meta_data['title'] = UtilData.get_file_name(yaml_path)
        MediaTableMaker.make_table(output_dir = output_dir, max_num_tr = max_num_tr, **meta_data)

    @staticmethod
    def make_table(
            output_dir:str = None,
            title:str = '',
            sub_title:str = '',
            audio_dir_meta_list:list = list(),
            audio_name_list:list = None,
            audio_name_list_ref_dir:str = None,
            sr:int = 44100,
            spec_type:Literal['mel', 'stft', 'x'] = 'mel',
            max_num_tr:int = 5,
            return_html:bool = False
        ):
        if output_dir is None: output_dir = f'./output/{title}'
        if audio_name_list is None and audio_name_list_ref_dir is not None:
            audio_name_list = [meta['file_name'] for meta in UtilData.walk(audio_name_list_ref_dir)]
            audio_name_list.sort()
        html_util = HTMLUtil(output_dir=output_dir)
        html_list = list()
        html_list.append(html_util.get_html_text(title))
        html_list.append(html_util.get_html_text(sub_title, tag='h2'))

        audio_html_args:dict = {
            'audio_dir_meta_list': audio_dir_meta_list,
            'sr': sr,
            'spec_type': spec_type,
            'max_num_tr': max_num_tr,
        }

        if isinstance(audio_name_list, list):
            html_list += MediaTableMaker.get_html_list(html_util=html_util, audio_name_list=audio_name_list, **audio_html_args)
        elif isinstance(audio_name_list, dict):
            for audio_catetory_name, audio_name_list in tqdm(audio_name_list.items(), desc='audio category'):
                html_list.append(html_util.get_html_text(audio_catetory_name, tag='h3'))
                html_list += MediaTableMaker.get_html_list(html_util=html_util, audio_name_list=audio_name_list, **audio_html_args)
        if return_html: return html_list
        else: html_util.save_html(html_list)
    
    @staticmethod
    def get_html_list(
        html_util:HTMLUtil,
        audio_dir_meta_list:list,
        audio_name_list:list,
        sr:int = 44100,
        spec_type:Literal['mel', 'stft', 'x'] = 'mel',
        max_num_tr:int = 5,
    ) -> list:
        html_list = list()
        html_dict_list = list()
        for audio_name in tqdm(audio_name_list, desc='audio list'):
            if len(html_dict_list) > max_num_tr:
                html_list += html_util.get_table_html_list(html_dict_list)
                html_dict_list = list()

            table_row_dict_audio = dict()
            table_row_dict_spec = dict()
            table_row_dict_audio['name'] = f'''<div style="width:{TD_WIDTH}px; overflow-wrap: break-word;">{audio_name}<div>'''
            for audio_dir_meta in audio_dir_meta_list:
                audio_dir_name = audio_dir_meta.get('name', audio_dir_meta['dir'].split('/')[-1])
                audio_path:str = MediaTableMaker.get_file_path(audio_name, audio_dir_meta)
                if os.path.isfile(audio_path):
                    img_path = None
                    spec_type_for_audio = spec_type
                    if 'img_dir' in audio_dir_meta:
                        img_path:str = MediaTableMaker.get_file_path(audio_name, audio_dir_meta, ext='png', dir_path=audio_dir_meta['img_dir'])
                        spec_type_for_audio = None
                    media_html_dict:dict = html_util.get_html_audio(audio_path = audio_path, sample_rate=sr, spec_type=spec_type_for_audio, spec_path=img_path)
                else:
                    file_strict:bool = audio_dir_meta.get('file_strict', True)
                    if file_strict: 
                        Util.print(f'File not found: {audio_path}', msg_type='error')
                        raise FileNotFoundError
                    else:
                        media_html_dict = {'audio': BLANK_COMPONENT, 'spec': BLANK_COMPONENT}
                table_row_dict_audio[audio_dir_name] = media_html_dict['audio']
                table_row_dict_spec[audio_dir_name] = media_html_dict.get('spec', BLANK_COMPONENT)
            html_dict_list.append(table_row_dict_audio)
            html_dict_list.append(table_row_dict_spec)
        
        if len(html_dict_list) > 0: html_list += html_util.get_table_html_list(html_dict_list)
        return html_list

    
    @staticmethod
    def get_file_path(audio_name:str, audio_dir_meta:dict, ext:Literal['wav', 'png'] = 'wav', dir_path:str = None) -> str:
        if audio_dir_meta.get('use_only_name', True): audio_name = audio_name.split('/')[-1]
        audio_name_pre_post_fix = audio_dir_meta.get('audio_name_pre_post_fix',['',''])
        audio_file_name:str = audio_dir_meta.get('audio_name', None)
        if audio_file_name is None:
            audio_file_name = f'{audio_name_pre_post_fix[0]}{audio_name}{audio_name_pre_post_fix[1]}.{ext}'
        else:
            audio_file_name = f'{audio_name}/{audio_name_pre_post_fix[0]}{audio_file_name}{audio_name_pre_post_fix[1]}.{ext}'
        audio_path:str = f"{audio_dir_meta['dir'] if dir_path is None else dir_path}/{audio_file_name}"
        
        if os.path.isfile(audio_path): return audio_path
        else: return unicodedata.normalize("NFC", audio_path)
