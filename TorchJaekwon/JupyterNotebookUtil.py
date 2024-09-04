from typing import List, Literal, Union, Tuple
try: import IPython.display as ipd
except:  print('[error] there is no IPython package')
try: import pandas as pd
except:  print('[error] there is no pandas package')

import re
import librosa

from TorchJaekwon.Util.UtilAudio import UtilAudio  
from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec 
from TorchJaekwon.Util.UtilData import UtilData

LOWER_IS_BETTER_SYMBOL = "↓"
HIGHER_IS_BETTER_SYMBOL = "↑"
PLUS_MINUS_SYMBOL = "±"

class JupyterNotebookUtil():
    def __init__(self,
                 output_dir:str = None,
                 table_data_width:int = None,
                 audio_sr:int = 44100
                 ) -> None:
        self.indent:str = '  '
        self.media_idx_dict:dict = {'audio':0, 'img':0}
        self.html_start_list:List[str] = [
            '<!DOCTYPE html>',
            '<head>',
            '<meta charset="utf-8" />',
            '<meta name="viewport" content="width=device-width, initial-scale=1" />',
            '<meta name="theme-color" content="#000000" />',
            '<style>',
            'h1, th {',
            'font-family: Arial, sans-serif;',
            '}',
            'td {',
            'font-family: Arial, sans-serif;',
            f'''width: {'fit-content' if table_data_width is None else table_data_width+'px'};''',
            '}',
            'table {',
            'border: 1px solid #444444;',
            'border-collapse: collapse;',
            '}',
            '.media-div {',
            'display: flex;',
            'align-items: center;',
            'justify-content: center;',
            '}',
            '</style>',
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

        mel_spec_config = UtilAudioMelSpec.get_default_mel_spec_config(audio_sr)
        self.mel_spec_util = UtilAudioMelSpec(**mel_spec_config)
    
    def get_table_html_list(self,
                            dict_list: List[dict],
                            use_pandas:bool = False
                            ) -> List[str]:
        '''
        Keys will be the table head items
        Values will be the table body items

        dict_list = [
        {'name':'test_sample_name', 'model1': html_code/float/str, 'model2': html_code/float/str, ...},
        /
        {'name':'model_name', 'metric1': html_code/float/str, 'metric2': html_code/float/str, ...},
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
                html_list.append(f'''<td><div class="media-div">{html_dict.get(table_head_item,'')}</div></td>''')
            html_list.append('</tr>')
        html_list.append('</tbody>')
        html_list.append('</table>')

        return html_list
    
    def save_html(self, html_list:List[str], file_name:str = 'plot.html') -> None:
        final_html_list:list = self.html_start_list + html_list + self.html_end_list
        indent_depth:int = 0
        for idx in range(1, len(final_html_list)):
            indent_depth += self.get_indent_depth_changed(final_html_list[idx - 1], final_html_list[idx])
            final_html_list[idx] = self.indent * indent_depth + final_html_list[idx]
        UtilData.txt_save(f'{self.output_dir}/{file_name}', final_html_list)
    
    def get_html_text(self, 
                      text:str,
                      tag:Literal['h1','h2','h3','h4','h5','h6','p'] = 'h1'
                      ) -> str:
        return f'<{tag}>{text}</{tag}>'
    
    def get_html_img(self,
                     src_path:str = None,
                     width:int=150
                    ) -> str: #html code
        style:str = '' if width is None else f'style="width:{width}px"'
        return f'''<img src="{src_path}" {style}/>'''
    
    def get_media_path(self, type:Literal['audio','img']) -> str:
        ext_dict = {'audio':'wav', 'img':'png'}
        path_dict = dict()
        path_dict['abs'] = f'{self.output_dir}/{self.media_save_dir_name}/{type}_{str(self.media_idx_dict[type]).zfill(5)}.{ext_dict[type]}'
        path_dict['relative'] = f'''./{self.media_save_dir_name}{path_dict['abs'].split(self.media_save_dir_name)[-1]}'''
        self.media_idx_dict[type] += 1
        return path_dict
    
    def get_html_audio(self,
                       audio_path:str = None,
                       cp_to_html_dir:bool = True,
                       sample_rate:int = None,
                       mel_spec_plot:bool = True,
                       spec_plot:bool = False,
                       width:int=200
                       ) -> Union[str, Tuple[str,str]]: #audio_html_code, img_html_code
        style:str = '' if width is None else f'style="width:{width}px"'
        if cp_to_html_dir:
            audio, sr = UtilAudio.read(audio_path = audio_path, sample_rate=sample_rate)
            path_dict = self.get_media_path('audio')
            UtilAudio.write(audio_path=path_dict['abs'], audio=audio, sample_rate=sr)
            audio_path = path_dict['relative']

        html_code_dict = dict()
        html_code_dict['audio'] = f'''<audio controls {style}> <source src="{audio_path}" type="audio/wav" /> </audio>'''
        if mel_spec_plot:
            mel_spec = self.mel_spec_util.get_hifigan_mel_spec(audio)
            if len(mel_spec.shape) == 3: mel_spec = mel_spec[0]
            img_path = f'{self.output_dir}/{self.media_save_dir_name}/img_{str(self.media_idx_dict["img"]).zfill(5)}.png'
            self.media_idx_dict["img"] += 1
            self.mel_spec_util.mel_spec_plot(save_path=img_path, mel_spec=mel_spec)
            img_path = f'./{self.media_save_dir_name}{img_path.split(self.media_save_dir_name)[-1]}'
            html_code_dict['mel'] = self.get_html_img(img_path, width)
        
        if spec_plot:
            stft_mag = self.mel_spec_util.stft_torch(audio)["mag"].squeeze()
            stft_db = librosa.amplitude_to_db(stft_mag)
            path_dict = self.get_media_path('img')
            self.mel_spec_util.mel_spec_plot(save_path=path_dict['abs'], mel_spec=stft_db)
            html_code_dict['spec'] = self.get_html_img(path_dict['relative'], width)
        
        return html_code_dict
    
    def get_html_tag_list(self, html_str:str) -> List[str]:
        html_tag_list = re.findall(r'</?[^>]+>', html_str)
        for idx in range(len(html_tag_list)):
            html_str_split = html_tag_list[idx].split(' ')
            if len(html_str_split) > 1:
                html_tag_list[idx] = html_str_split[0] + html_str_split[-1]
        return html_tag_list
    
    def get_indent_depth_changed(self, prev_str:str, current_str:str) -> bool:
        prev_tag_list = self.get_html_tag_list(prev_str)
        current_tag_list = self.get_html_tag_list(current_str)
        if len(current_tag_list) == 0 or len(prev_tag_list) == 0:
            return 0
        
        if prev_tag_list[0] == current_tag_list[0]:
            return 0

        if '</' in current_tag_list[0]:
            return -1
        
        for prev_tag in prev_tag_list: 
            if '/' in prev_tag: 
                return 0
        
        if len(prev_tag_list) < 2 and not '</' in prev_tag_list[0]:
            return 1
        return 0
    
    @staticmethod
    def display_html_list(html_list:list) -> None:
        for html_result in html_list:
            ipd.display(ipd.HTML(html_result))
    