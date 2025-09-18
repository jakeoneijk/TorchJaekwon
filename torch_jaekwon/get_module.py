from typing import Literal, Callable, Union

import os
import importlib

from torch_jaekwon.path import TORCH_JAEKWON_PATH, CLASS_DIRS

def get_module_tj(
    root_path:str = None, # if class_type is not None, don't need to input root_path
    class_type:Literal['preprocessor', 'model', 'trainer', 'pytorch_dataset', 'lr_scheduler', 'loss', 'inferencer', 'evaluator'] = None,
    class_meta:dict = dict()
) -> Callable:
    if 'name' not in class_meta and isinstance(class_meta, dict):
        return {k: get_module_tj(root_path=root_path, class_type=class_type, class_meta=v) for k,v in class_meta.items()}
    else:
        return GetModule.get_module(root_path=root_path, class_type=class_type, module_name=class_meta.get('name'), arg_dict=class_meta.get('args', dict()))

class GetModule:
    @staticmethod
    def get_module(
        root_path:str = None, # if class_type is not None, don't need to input root_path
        class_type:Literal['preprocessor', 'model', 'trainer', 'pytorch_dataset', 'lr_scheduler', 'loss', 'inferencer', 'evaluator'] = None,
        module_name:tuple = None,
        arg_dict:dict = dict(),
    ) -> Callable:
        module_class:type = GetModule.get_module_class(root_path, class_type, module_name)
        module_instance = module_class(**arg_dict)
        return module_instance
    
    @staticmethod
    def get_module_class(
        root_path:str = None, # if class_type is not None, don't need to input root_path
        class_type:Literal['preprocessor', 'model', 'trainer', 'pytorch_dataset', 'lr_scheduler', 'loss', 'inferencer', 'evaluator'] = None,
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