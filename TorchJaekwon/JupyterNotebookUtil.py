from typing import List
try:    
    import IPython.display as ipd
except:
    print('[error] there is no ipython package')
import pandas as pd

class JupyterNotebookUtil():
    def __init__(self) -> None:
        self.lower_is_better_symbol:str = "↓"
        self.higher_is_better_symbol:str = "↑"

    @staticmethod
    def get_html_from_src_path(type:str,path:str,width:int=100) -> str:
        if type == "audio":
            if width is not None:
                return f"""<audio controls style='width:{width}px'><source src="{path}" type="audio/wav"></audio></td>"""
            else:
                return f"""<audio controls><source src="{path}" type="audio/wav"></audio></td>"""
        elif type == "img":
            if width is not None:
                return f"""<img src="{path}" style='width:{width}px'>"""
            else:
                return f"""<img src="{path}">"""
    
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
        {'name':'model1','metric1':html_code, 'metric2':html_code, ...}
        {'name':'model2','metric1':html_code, 'metric2':html_code, ...}
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