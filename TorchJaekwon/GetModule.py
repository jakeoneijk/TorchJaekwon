from typing import Optional

import os
import importlib

class GetModule:
    def __init__(self) -> None:
        '''
        self.root_path_dict:dict[str,str] = dict()
        self.root_path_dict["batch_sampler"] = "./Data/PytorchDataLoader/BatchSampler"
        self.root_path_dict["process"] = "./DataProcess/Process"
        self.root_path_dict["model"] = "./Model"
        self.root_path_dict["loss_function"] = "./Train/Loss/LossFunction"
        self.root_path_dict["evaluater"] = "./Evaluater"

        self.preprocess_realtime_root_path:str = "./PreprocessRealTime"
        '''
        pass
    @staticmethod
    def get_import_path_of_module(root_path:str, module_name:str ) -> Optional[str]:
        root_path_list:list = [root_path]
        if os.path.isdir(root_path.replace("./","./TorchJAEKWON/")):
            root_path_list.append(root_path.replace("./","./TorchJAEKWON/"))
        
        for root_path in root_path_list:
            for root,dirs,files in os.walk(root_path):
                if len(files) > 0:
                    for file in files:
                        if os.path.splitext(file)[0] == module_name:
                            return f'{root}/{os.path.splitext(file)[0]}'.replace("./","").replace("/",".")
        return None
    @staticmethod
    def get_module(root_path:str,module_name:str,module_arg=None,arg_unpack=False) -> object:
        module_path:str = GetModule.get_import_path_of_module(root_path,module_name)
        module_from = importlib.import_module(module_path)
        module_import_class = getattr(module_from,module_name)
        if module_arg is not None:
            module = module_import_class(**module_arg) if arg_unpack else module_import_class(module_arg)
        else:
            module = module_import_class()
        
        return module
    
    @staticmethod
    def get_module_class(root_path:str,module_name:str):
        module_path:str = GetModule.get_import_path_of_module(root_path,module_name)
        module_from = importlib.import_module(module_path)
        return getattr(module_from,module_name)
    
    @staticmethod
    def get_model(
        model_name:str,
        root_path:str = './Model'
        ):
        
        module_file_path:str = GetModule.get_import_path_of_module(root_path, model_name)
        file_module = importlib.import_module(module_file_path)
        class_module = getattr(file_module,model_name)
        model_parameter:dict = class_module.get_argument_of_this_model()
        model = class_module(**model_parameter)
        return model