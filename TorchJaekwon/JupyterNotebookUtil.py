from typing import List, Literal
try: import IPython.display as ipd
except:  print('[error] there is no IPython package')
import pandas as pd

class JupyterNotebookUtil():
    def __init__(self) -> None:
        self.lower_is_better_symbol:str = "↓"
        self.higher_is_better_symbol:str = "↑"
        self.plus_minus_symbol:str = "±"

    @staticmethod
    def get_html(type:Literal['audio','img'],
                 src_path:str,
                 width:int=100
                 ) -> str: #html code
        style:str = '' if width is None else f'style="width:{width}px"'
        if type == "audio":
            return f'''<audio controls {style}><source src="{src_path}" type="audio/wav" /></audio>''' #
        elif type == "img":
            return f'''<img src="{src_path}" {style}/>'''
    
    @staticmethod
    def dict_list_to_html(dict_list: List[dict]) -> str:
        '''
        dict_list = [
        {'name':'testaudio','audio1':html_code, 'audio2':html_code, 'image1':html_code ...}
        {'name':'testaudio','audio1':html_code, 'audio2':html_code, 'image1':html_code ...}
        ...
        ]
        '''
        df = pd.DataFrame(dict_list)
        return df.to_html(escape=False,index=False)
    
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