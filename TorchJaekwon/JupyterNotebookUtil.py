from typing import List, Literal, Union
try: import IPython.display as ipd
except:  print('[error] there is no IPython package')
try: import pandas as pd
except:  print('[error] there is no pandas package')
try: import numpy as np
except:  print('[error] there is no numpy package')
try: import torch
except:  print('[error] there is no torch package')

import re

from TorchJaekwon.Util.UtilAudio import UtilAudio   
from TorchJaekwon.Util.UtilData import UtilData

LOWER_IS_BETTER_SYMBOL = "↓"
HIGHER_IS_BETTER_SYMBOL = "↑"
PLUS_MINUS_SYMBOL = "±"

class JupyterNotebookUtil():
    def __init__(self, output_dir:str = None) -> None:
        self.indent:str = '  '
        self.media_idx_dict:dict = {'audio':0}
        self.html_start_list:List[str] = [
            '<!DOCTYPE html>',
            '<head>',
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            '<meta name="theme-color" content="#000000" />',
            '</head>',
            '<body>',
            '<div id="root">',
        ]
        self.html_end_list:List[str] = [
            '</div>',
            '</body>',
            '</html>',
        ]
        self.output_dir:str = output_dir
        self.media_save_dir_name:str = 'media'
    
    def dict_list_to_html_list(self, dict_list: List[dict], use_pandas:bool = False) -> List[str]:
        '''
        dict_list = [
        {'name':'testaudio','audio1':html_code, 'audio2':html_code, 'image1':html_code ...}
        {'name':'testaudio','audio1':html_code, 'audio2':html_code, 'image1':html_code ...}
        ...
        ]
        '''
        if use_pandas:
            df = pd.DataFrame(dict_list)
            return df.to_html(escape=False,index=False)
        
        html_list = list()
        table_head_item_list = list(dict_list[0].keys())
        html_list.append('<table border="1">')
        html_list.append('<thead>')
        html_list.append('<tr>')
        for table_head_item in table_head_item_list:
            html_list.append(f'<th>{table_head_item}</th>')
        html_list.append('</tr>')
        html_list.append('</thead>')

        html_list.append('<tbody>')
        for html_dict in dict_list:
            html_list.append('<tr>')
            for table_head_item in table_head_item_list:
                html_list.append(f'<td>{html_dict[table_head_item]}</td>')
            html_list.append('</tr>')
        html_list.append('</tbody>')
        html_list.append('</table>')

        return html_list
    
    def get_html_text(self, 
                      text:str,
                      tag:Literal['h1','h2','h3','h4','h5','h6','p'] = 'h1'
                      ) -> str:
        return f'<{tag}>{text}</{tag}>'
    
    def get_html_audio(self,
                       audio_path:str = None,
                       cp_to_html_dir:bool = True,
                       sample_rate:int = None,
                       width:int=200
                       ) -> str:
        style:str = '' if width is None else f'style="width:{width}px"'
        if cp_to_html_dir:
            audio, _ = UtilAudio.read(audio_path = audio_path, sample_rate=sample_rate)
            audio_path = f'{self.output_dir}/{self.media_save_dir_name}/audio_{str(self.media_idx_dict["audio"]).zfill(3)}.wav'
            self.media_idx_dict["audio"] += 1
            UtilAudio.write(audio_path, audio, sample_rate)
            audio_path = f'./{self.media_save_dir_name}{audio_path.split(self.media_save_dir_name)[-1]}'
        return f'''<audio controls {style}><source src="{audio_path}" type="audio/wav" /></audio>'''
    
    def get_html_tag_list(self, html_str:str) -> List[str]:
        html_tag_list = re.findall(r'</?[^>]+>', html_str)
        for idx in range(len(html_tag_list)):
            html_str_split = html_tag_list[idx].split(' ')
            if len(html_str_split) > 1:
                html_tag_list[idx] = html_str_split[0] + html_str_split[-1]
        return html_tag_list
    
    def save_html(self, html_list:List[str], file_name:str = 'plot.html') -> None:
        final_html_list:list = self.html_start_list + html_list + self.html_end_list
        indent_depth:int = 0
        for idx in range(1, len(final_html_list)):
            prev_tag_list = self.get_html_tag_list(final_html_list[idx - 1])
            current_tag_list = self.get_html_tag_list(final_html_list[idx])
            if prev_tag_list[0] != current_tag_list[0]:
                if '</' in current_tag_list[0]:
                    indent_depth -= 1
                elif len(prev_tag_list) < 2 and not '</' in prev_tag_list[0]:
                    indent_depth += 1
                
            final_html_list[idx] = self.indent * indent_depth + final_html_list[idx]
        UtilData.txt_save(f'{self.output_dir}/{file_name}', final_html_list)
        
    def get_html_media(type:Literal['audio','img'],
                       src_path:str = None,
                       data: dict = None, # {'data': Union[ndarray, Tensor], 'meta_data': { 'sample_rate': int }}
                       width:int=150
                       ) -> str: #html code
        assert src_path is not None or data is not None, '[Error] src_path or data must be not None'
        if data is not None:
            if type == "audio":
                return f'''<audio controls {style}><source src="data:audio/wav;base64,{data}" /></audio>'''

        style:str = '' if width is None else f'style="width:{width}px"'
        if type == "audio":
            return f'''<audio controls {style}><source src="{src_path}" type="audio/wav" /></audio>''' #
        elif type == "img":
            return f'''<img src="{src_path}" {style}/>'''
    
    @staticmethod
    def display_html_list(html_list:list) -> None:
        for html_result in html_list:
            ipd.display(ipd.HTML(html_result))
    
    @staticmethod
    def html_table_from_dict_list(dict_list: List[dict]):
        '''
        dict_list = [
        {'name':'model1','metric1':float, 'metric2':float, ...}
        {'name':'model2','metric1':float, 'metric2':float, ...}
        ...
        ]
        '''
        html:str = ''
        new_line:str = '\n'
        
        html += '<table border="1">' + new_line

        header_list = list(dict_list[0].keys())
        for header_name in header_list:
            html += f'<th>{header_name}</th>{new_line}'
        
        for body_dict in dict_list:
            html += f'<tr>{new_line}'
            for header_name in header_list:
                html += f'<td>{body_dict[header_name]}</td>{new_line}'
            html += f'</tr>{new_line}'

        html+= '</table>'
        return html