from typing import Literal, Callable

import os
import importlib

from TorchJaekwon.Path import TORCH_JAEKWON_PATH, CLASS_DIR_PATH_DICT
from TorchJaekwon.Util import Util

try: import torch.nn as nn
except: print('''Can't import torch.nn''')

class GetModule:
    @staticmethod
    def get_import_path_of_module(
        root_path:str, 
        module_name:str,
    ) -> str:
        root_path_list:list = [root_path, root_path.replace("./",f'{TORCH_JAEKWON_PATH}/')]
        torch_jaekwon_parent_path:str = '/'.join(TORCH_JAEKWON_PATH.split('/')[:-1])
        
        for root_path in root_path_list:
            for root, _, files in os.walk(root_path):
                for file in files:
                    file_name:str = os.path.splitext(file)[0]
                    if file_name == module_name:
                        parent_path_to_be_eliminated:str = f'{torch_jaekwon_parent_path}/' if TORCH_JAEKWON_PATH in root else './'
                        return f'{root}/{file_name}'.replace(parent_path_to_be_eliminated,"").replace("/",".")
        return None
    
    @staticmethod
    def get_module_class(
        root_path:str = None,
        class_type:Literal['lr_scheduler'] = None,
        module_name:str = None,
    ):
        if class_type is not None:
            root_path = CLASS_DIR_PATH_DICT[class_type]
        module_path:str = GetModule.get_import_path_of_module(root_path,module_name)
        module_from = importlib.import_module(module_path)
        return getattr( module_from, module_name )
    
    @staticmethod
    def get_model(
        model_name:str,
        root_path:str = './Model'
    ) -> nn.Module:
        module_file_path:str = GetModule.get_import_path_of_module(root_path, model_name)
        file_module = importlib.import_module(module_file_path)
        class_module = getattr(file_module,model_name)
        argument_getter:Callable[[],dict] = getattr(class_module,'get_argument_of_this_model',lambda: dict())
        model_parameter:dict = argument_getter()
        if not model_parameter:
            try: 
                from HParams import HParams
                model_parameter = HParams().model.class_meta_dict.get(model_name,{})
                if not model_parameter: 
                    Util.print(f'''[GetModule] Model [{model_name}] doesn't have changed arguments''', 'info')
            except: 
                Util.print(f'''[GetModule] Model [{model_name}] doesn't have changed arguments''', 'info')
        model:nn.Module = class_module(**model_parameter)
        return model