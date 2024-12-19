import os
from tqdm import tqdm
from TorchJaekwon.Util.Util import Util 
from TorchJaekwon.Util.UtilData import UtilData
from TorchJaekwon.Util.TableMaker import JupyterNotebookUtil

class AudioListeningTableMaker:
    @staticmethod
    def get_yaml_example(output_dir:str = './') -> None:
        file_path:str = f'{os.path.dirname(__file__)}/audio_table_config_example.yaml'
        Util.system(f'cp {file_path} {output_dir}')

    @staticmethod
    def make_table_from_config_path(yaml_path:str, output_dir:str = None) -> None:
        meta_data:dict = UtilData.yaml_load(yaml_path)
        AudioListeningTableMaker.make_table(output_dir = output_dir, **meta_data)
        

    @staticmethod
    def make_table(
            title:str = '',
            sub_title:str = '',
            audio_dir_meta_list:list = list(),
            audio_name_list:list = None,
            output_dir:str = None,
            sr:int = 44100,
            max_num_tr:int = 5,
            return_html:bool = False
        ):
        if output_dir is None: output_dir = f'./output/{title}'
        html_util = JupyterNotebookUtil(output_dir=output_dir)
        html_list = list()
        html_list.append(html_util.get_html_text(title))
        html_list.append(html_util.get_html_text(sub_title, tag='h3'))

        html_dict_list = list()
        for audio_name in tqdm(audio_name_list, desc='audio list'):
            table_row_dict_audio = dict()
            table_row_dict_spec = dict()
            table_row_dict_audio['name'] = f'''<div style="width:300px">{audio_name}<div>'''
            for audio_dir_meta in audio_dir_meta_list:
                audio_dir_name = audio_dir_meta.get('name', audio_dir_meta['dir'].split('/')[-1])
                media_html_dict:dict = html_util.get_html_audio(audio_path = AudioListeningTableMaker.get_audio_path(audio_name, audio_dir_meta), sample_rate=sr)
                table_row_dict_audio[audio_dir_name] = media_html_dict['audio']
                table_row_dict_spec[audio_dir_name] = media_html_dict['mel']
            html_dict_list.append(table_row_dict_audio)
            html_dict_list.append(table_row_dict_spec)
            if len(html_dict_list) >= max_num_tr:
                html_list += html_util.get_table_html_list(html_dict_list)
                html_dict_list = list()
        
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
        return f"{audio_dir_meta['dir']}/{audio_file_name}"
