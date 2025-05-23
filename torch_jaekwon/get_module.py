from typing import Literal, Callable, Union

import os
import importlib

from torch_jaekwon.path import TORCH_JAEKWON_PATH, CLASS_DIRS
from torch_jaekwon.util import Util

try: import torch.nn as nn
except: print('''Can't import torch.nn''')

class GetModule:
    @staticmethod
    def get_module_class(
        root_path:str = None, # if class_type is not None, don't need to input root_path
        class_type:Literal['preprocessor', 'trainer', 'data_loader', 'pytorch_dataset', 'lr_scheduler', 'loss'] = None,
        module_name:Union[str,tuple] = None, # if str: module_name==file_name==class_name. if tuple: module_name[0]==file_name, module_name[1]==class_name
    ) -> type:
        if class_type is not None:
            root_path = getattr(CLASS_DIRS, class_type)

        if isinstance(module_name, str):
            file_name:str = module_name
            class_name:str = module_name
        elif isinstance(module_name, tuple) or (isinstance(module_name, list) and len(module_name) == 2):
            file_name:str = module_name[0]
            class_name:str = module_name[1]
        else:
            raise ValueError(f'''[GetModule] module_name should be str or tuple. {module_name}''')
        
        module_path:str = GetModule.get_import_path_of_module(root_path, file_name)
        module_from = importlib.import_module(module_path)
        return getattr(module_from, class_name)
    
    @staticmethod
    def get_model(
        module_name:Union[tuple, list] = None, # if str: module_name==file_name==class_name. if tuple: module_name[0]==file_name, module_name[1]==class_name
        root_path:str = './model'
    ) -> nn.Module:
        assert isinstance(module_name, (tuple, list)), f'''[GetModule] module_name should be tuple or list. {module_name}'''
        class_module = GetModule.get_module_class(root_path = root_path, module_name = module_name)
        argument_getter:Callable[[],dict] = getattr(class_module,'get_argument_of_this_model',lambda: dict())
        model_parameter:dict = argument_getter()
        if not model_parameter:
            try: 
                from h_params import HParams
                model_parameter = HParams().model.class_meta_dict.get(module_name[1],{})
                if not model_parameter: 
                    Util.print(f'''[GetModule] Model [{module_name}] doesn't have changed arguments''', 'info')
            except: 
                Util.print(f'''[GetModule] Model [{module_name}] doesn't have changed arguments''', 'info')
        model:nn.Module = class_module(**model_parameter)
        return model
    
    @staticmethod
    def get_import_path_of_module(
        root_path:str, 
        file_name:str,
    ) -> str:
        root_path_list:list = [root_path, root_path.replace("./",f'{TORCH_JAEKWON_PATH}/')]
        torch_jaekwon_parent_path:str = '/'.join(TORCH_JAEKWON_PATH.split('/')[:-1])
        
        for root_path in root_path_list:
            for root, _, files in os.walk(root_path):
                for file in files:
                    if file_name == os.path.splitext(file)[0]:
                        parent_path_to_be_eliminated:str = f'{torch_jaekwon_parent_path}/' if TORCH_JAEKWON_PATH in root else './'
                        return f'{root}/{file_name}'.replace(parent_path_to_be_eliminated,"").replace("/",".")
        return None