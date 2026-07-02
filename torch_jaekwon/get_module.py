from typing import Callable

import importlib

def get_module_tj(
    class_meta:dict = dict()
) -> Callable:
    if 'path' not in class_meta and isinstance(class_meta, dict):
        return {k: get_module_tj(class_meta=v) for k,v in class_meta.items()}
    else:
        return GetModule.get_module(module_name=class_meta.get('path'), arg_dict=class_meta.get('args', dict()))

class GetModule:
    @staticmethod
    def get_module(
        module_name:str = None,
        arg_dict:dict = dict(),
    ) -> Callable:
        module_class:type = GetModule.get_module_class(module_name)
        module_instance = module_class(**arg_dict)
        return module_instance

    @staticmethod
    def get_module_class(
        module_name:str = None, # fully-qualified dotted import path: '<module>.<ClassName>' (e.g. 'model.unet.UNet', 'torch_jaekwon.train.trainer.trainer.Trainer')
    ) -> type:
        if not isinstance(module_name, str) or '.' not in module_name:
            raise ValueError(f"[GetModule] module_name must be a fully-qualified dotted path '<module>.<ClassName>', got {module_name!r}")
        module_path, class_name = module_name.rsplit('.', 1)
        module_from = importlib.import_module(module_path)
        return getattr(module_from, class_name)
