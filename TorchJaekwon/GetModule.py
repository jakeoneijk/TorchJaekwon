from typing import Optional, Callable

import os
import importlib

try:
    import torch.nn as nn
except:
    print('''Can't import torch.nn''')

from HParams import HParams, TORCH_JAEKWON_PATH

class GetModule:
    @staticmethod
    def get_import_path_of_module(root_path:str, module_name:str) -> Optional[str]:
        root_path_list:list = [root_path]
        torch_jaekwon_path = f'{TORCH_JAEKWON_PATH}/TorchJaekwon/'
        if os.path.isdir(root_path.replace("./",torch_jaekwon_path)):
            root_path_list.append(root_path.replace("./",torch_jaekwon_path))
        
        for root_path in root_path_list:
            for root,dirs,files in os.walk(root_path):
                if len(files) > 0:
                    for file in files:
                        if os.path.splitext(file)[0] == module_name:
                            if torch_jaekwon_path in root:
                                return f'{root}/{os.path.splitext(file)[0]}'.replace(TORCH_JAEKWON_PATH+'/','').replace("/",".")
                            else:
                                return f'{root}/{os.path.splitext(file)[0]}'.replace("./","").replace("/",".")
        return None
    
    @staticmethod
    def get_module_class(root_path:str,module_name:str):
        module_path:str = GetModule.get_import_path_of_module(root_path,module_name)
        module_from = importlib.import_module(module_path)
        return getattr(module_from,module_name)
    
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
        if len(model_parameter) == 0:
            model_parameter = HParams().model.class_meta_dict.get(model_name,{})
        if not model_parameter: 
            model_parameter = getattr(HParams().model,model_name,dict())
            if not model_parameter: print(f'''GetModule: Model [{model_name}] doesn't have changed arguments''')
        model:nn.Module = class_module(**model_parameter)
        return model