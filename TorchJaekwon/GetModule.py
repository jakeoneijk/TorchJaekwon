from typing import Literal, Callable

import os
import importlib

from TorchJaekwon.Path import TORCH_JAEKWON_PATH, CLASS_DIR_PATH_DICT

try: import torch.nn as nn
except: print('''Can't import torch.nn''')
try: 
    from HParams import HParams
    hparams_exist = True
except: 
    hparams_exist = False
    print('There is no Hparams')

class GetModule:
    @staticmethod
    def get_import_path_of_module(
        root_path:str, 
        module_name:str,
    ) -> str:
        root_path_list:list = [root_path]
        root_path_list.append(root_path.replace("./",f'{TORCH_JAEKWON_PATH}/'))
        
        for root_path in root_path_list:
            for root,dirs,files in os.walk(root_path):
                if len(files) > 0:
                    for file in files:
                        if os.path.splitext(file)[0] == module_name:
                            if TORCH_JAEKWON_PATH in root:
                                torch_jaekwon_parent_path:str = '/'.join(TORCH_JAEKWON_PATH.split('/')[:-1])
                                return f'{root}/{os.path.splitext(file)[0]}'.replace(torch_jaekwon_parent_path+'/','').replace("/",".")
                            else:
                                return f'{root}/{os.path.splitext(file)[0]}'.replace("./","").replace("/",".")
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
        if hparams_exist:
            if len(model_parameter) == 0:
                model_parameter = HParams().model.class_meta_dict.get(model_name,{})
            if not model_parameter: 
                model_parameter = getattr(HParams().model,model_name,dict())
                if not model_parameter: print(f'''GetModule: Model [{model_name}] doesn't have changed arguments''')
        model:nn.Module = class_module(**model_parameter)
        return model