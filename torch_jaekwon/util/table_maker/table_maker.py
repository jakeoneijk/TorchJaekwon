from typing import Literal, Callable

import os
import unicodedata
from tqdm import tqdm
from .. import util, util_data
from . import HTMLUtil

TD_WIDTH = 300
BLANK_COMPONENT = f'<div style="width:{TD_WIDTH}px"> X <div>'
NAME_TAG = 'name'

class TableMaker:
    @staticmethod
    def get_yaml_example(output_dir:str = './') -> None:
        file_path:str = f'{os.path.dirname(__file__)}/table_config_example.yaml'
        util.system(f'cp {file_path} {output_dir}')

    @staticmethod
    def make_table_from_config_path(
        yaml_path:str, 
        output_dir:str = None, 
        max_num_tr:int = 5,
        get_item:Callable = lambda meta_data: {'item':None, 'type':None},
        get_data_name_list:Callable = None, #lambda meta_data: []
        get_file_path:Callable = None, #lambda (data_name, model_meta, ext): str
    ) -> None:
        meta_data:dict = util_data.yaml_load(yaml_path)
        if not meta_data.get('title',None): meta_data['title'] = util_data.get_file_name(yaml_path)
        TableMaker.make_table(output_dir = output_dir, max_num_tr = max_num_tr, get_item = get_item, get_data_name_list=get_data_name_list, get_file_path=get_file_path, **meta_data)

    @staticmethod
    def make_table(
        output_dir:str = None,
        title:str = '',
        sub_title:str = '',
        model_meta_list:list = list(),
        data_name_list:list = None,
        data_name_list_ref_dir:str = None,
        max_num_tr:int = 5,
        return_html:bool = False,
        transpose:bool = False,
        get_item:Callable = lambda meta_data: {'item':None, 'type':None},
        get_data_name_list:Callable = None, #lambda meta_data: []
        get_file_path:Callable = None, #lambda (data_name, model_meta, ext): str
        audio_config:dict = None,
    ) -> None:
        if output_dir is None: output_dir = f'./output/{title}'
        if get_data_name_list is not None:
            data_name_list = get_data_name_list({'data_name_list_ref_dir': data_name_list_ref_dir})
        elif data_name_list is None and data_name_list_ref_dir is not None:
            data_name_list = [meta['file_name'] for meta in util_data.walk(data_name_list_ref_dir)]
            data_name_list.sort()
        html_util = HTMLUtil(output_dir=output_dir)
        html_list = list()
        html_list.append(html_util.get_html_text(title))
        html_list.append(html_util.get_html_text(sub_title, tag='h2'))

        if isinstance(data_name_list, list): data_name_list = {'': data_name_list}
        get_str_html = lambda x: f'<div style="width:{TD_WIDTH}px; overflow-wrap: break-word;">{x}<div>'
        for case_name, data_name_list in tqdm(data_name_list.items(), desc='data category'):
            html_dict_list = list()
            if case_name: html_list.append(html_util.get_html_text(case_name, tag='h3'))
            for data_name in tqdm(data_name_list, desc='data'):
                if len(html_dict_list) > max_num_tr:
                    html_list += TableMaker.get_table_html_list(html_dict_list, transpose=transpose)
                    html_dict_list = list()
                table_row_dict_list = [dict()] #[{'first_row_model1':, ''first_row_model2''}, {'second_row_model1': ...} ...]
                table_row_dict_list[0][NAME_TAG] = get_str_html(data_name)
                for model_meta in model_meta_list:
                    model_name = get_str_html(model_meta.get(NAME_TAG, model_meta['dir'].split('/')[-1]))
                    ext = model_meta.get('ext', 'wav')
                    item_dict = dict()
                    try:
                        if ext == 'function':
                            item_dict:dict = get_item({'model_meta': model_meta, 'data_name': data_name, 'case_name': case_name})
                            item = item_dict.get('item', None)
                            item_type:str = item_dict.get('type', None)
                            if item_type is None:
                                html_code_list = [item]
                            else:
                                raise NotImplementedError(f"item type '{item_type}' is not implemented.")
                        if ext == 'wav':
                            #audio_path:str = TableMaker.get_default_file_path(media_name, comparison_meta)
                            #img_path = None
                            #spec_type_for_audio = spec_type
                            #if 'img_dir' in comparison_meta:
                            #    img_path:str = TableMaker.get_default_file_path(media_name, comparison_meta, ext='png', dir_path=comparison_meta['img_dir'])
                            #    spec_type_for_audio = None
                            #media_html_dict:dict = html_util.get_html_audio(audio_path = audio_path, sample_rate=sr, spec_type=spec_type_for_audio, spec_path=img_path)
                            #table_row_dict_audio[comparison_name] = media_html_dict['audio']
                            #table_row_dict_spec[comparison_name] = media_html_dict.get('spec', BLANK_COMPONENT)
                            raise NotImplementedError(f"ext '{ext}' is not implemented.")
                        if ext in ['mp4']:
                            file_path:str = get_default_file_path(data_name, model_meta, ext) if get_file_path is None else get_file_path(data_name, model_meta, ext)
                            html_code_list:list = html_util.get_html_video(file_path=file_path)
                    except Exception as e:
                        file_strict:bool = model_meta.get('file_strict', True)
                        if file_strict:
                            util.log(f'File not found: {audio_path}', msg_type='error')
                            raise FileNotFoundError
                        else:
                            html_code_list = [BLANK_COMPONENT]
                    for i, html_code in enumerate(html_code_list):
                        table_row_dict_list[i][model_name] = html_code
                html_dict_list += table_row_dict_list
            if len(html_dict_list) > 0:
                html_list += TableMaker.get_table_html_list(html_dict_list, transpose=transpose)

        if return_html: return html_list
        else: html_util.save_html(html_list)
    
    @staticmethod
    def get_table_html_list(html_dict_list:list, transpose:bool) -> list:
        if transpose:
            html_dict_list_t = list()
            comparison_name_list:list = [comparison_name for comparison_name in list(html_dict_list[0].keys()) if comparison_name != NAME_TAG]
            for comparison_name in comparison_name_list:
                html_dict_t = {NAME_TAG: comparison_name}
                for html_dict in html_dict_list:
                    html_dict_t[html_dict[NAME_TAG]] = html_dict[comparison_name]
                html_dict_list_t.append(html_dict_t)
            html_dict_list = html_dict_list_t
        return HTMLUtil.get_table_html_list(html_dict_list)
    
def get_default_file_path(file_name:str, model_meta:dict, ext:str = 'wav', dir_path:str = None) -> str:
    if model_meta.get('use_only_name', True): file_name = file_name.split('/')[-1]
    file_name_pre_post_fix = model_meta.get('file_name_pre_post_fix',['',''])
    audio_file_name:str = model_meta.get('file_name', None)
    if audio_file_name is None:
        audio_file_name = f'{file_name_pre_post_fix[0]}{file_name}{file_name_pre_post_fix[1]}.{ext}'
    else:
        audio_file_name = f'{file_name}/{file_name_pre_post_fix[0]}{audio_file_name}{file_name_pre_post_fix[1]}.{ext}'
    audio_path:str = f"{model_meta['dir'] if dir_path is None else dir_path}/{audio_file_name}"
    
    if os.path.isfile(audio_path): return audio_path
    else: return unicodedata.normalize("NFC", audio_path)
