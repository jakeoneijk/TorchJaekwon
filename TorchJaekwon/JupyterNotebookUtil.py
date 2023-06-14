from typing import List

import IPython.display as ipd
import pandas as pd

class JupyterNotebookUtil():
    def __init__(self) -> None:
        self.lower_is_better_symbol:str = "↓"
        self.higher_is_better_symbol:str = "↑"

    def get_html_from_src_path(self,type:str,path:str,width:int=200) -> str:
        if type == "audio":
            if width is not None:
                return f"""<audio controls style='width:{width}px'><source src="{path}" type="audio/wav"></audio></td>"""
            else:
                return f"""<audio controls><source src="{path}" type="audio/wav"></audio></td>"""
        elif type == "img":
            return f"""<img src="{path}">"""
    
    def dict_list_to_html(self,pandas_list: List[dict]) -> str:
        '''
        panda_list = [
        {'name':'testaudio','audio':html_code}
        ...
        ]
        '''
        df = pd.DataFrame(pandas_list)
        return df.to_html(escape=False,index=False)
    
    def display_html_list(self,html_list:list) -> None:
        for html_result in html_list:
            ipd.display(ipd.HTML(html_result))
    