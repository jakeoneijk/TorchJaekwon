#type
from typing import Type, Literal, Dict, List, Union, Literal

#package
import os
import argparse
import numpy as np

#torchjaekwon
from .get_module import GetModule
from .util import Util
from .path import ARTIFACTS_DIRS

#internal
from h_params import HParams

class Controller():
    def __init__(self) -> None:
        self.set_argparse()

        self.config_name:str = HParams().mode.config_name
        self.stage: Literal['preprocess', 'train', 'inference', 'evaluate'] = HParams().mode.stage

        self.config_per_dataset_dict: Dict[str, dict] = HParams().data.config_per_dataset_dict
        self.eval_class_meta:dict = HParams().evaluate.class_meta # {'name': 'Evaluater', 'args': {}}

    def run(self) -> None:
        print("=============================================")
        print(f"{self.stage} start.")
        print("=============================================")
        print(f"{self.config_name} start.")
        print("=============================================")
        
        getattr(self,self.stage)()
        
        print("Finish app.")

    def preprocess(self) -> None:
        from .data.preprocess.preprocessor import Preprocessor
        for data_name in self.config_per_dataset_dict:
            for preprocessor_meta in self.config_per_dataset_dict[data_name]['preprocessor_class_meta_list']:
                preprocessor_class_name:str = preprocessor_meta['name']
                preprocessor_args:dict = {
                    'data_name': data_name,
                    'num_workers': HParams().resource.num_workers,
                    'device': HParams().resource.device,
                }
                preprocessor_args.update(preprocessor_meta['args'])

                preprocessor_class:Type[Preprocessor] = GetModule.get_module_class( 
                    class_type='preprocessor', 
                    module_name = preprocessor_class_name 
                )
                preprocessor:Preprocessor = preprocessor_class(**preprocessor_args)                             
                preprocessor.preprocess_data()                           

    def train(self) -> None:
        import torch
        from torch_jaekwon.train.trainer.trainer import Trainer
        from torch_jaekwon.train.logger.logger import Logger

        logger = Logger(
            experiment_name = HParams().mode.config_name,
            use_time_on_experiment_name = False,
            project_name = HParams().log.project_name,
            visualizer_type = HParams().log.log_tool,
            root_dir_path = f'{ARTIFACTS_DIRS.train}/{HParams().mode.config_name}',
            is_resume = HParams().mode.is_train_resume,
        )

        trainer_args = {
            # data
            'data_class_meta_dict': HParams().dataloader,
            # model
            'model_class_meta_dict': HParams().model.class_meta,
            # loss
            'loss_meta_dict': getattr(HParams().train, 'loss', None),
            # optimizer
            'optimizer_class_meta_dict': HParams().train.optimizer['class_meta'],
            'optimizer_step_unit': getattr(HParams().train,'optimizer_step_unit',1),
            'lr_scheduler_class_meta_dict': HParams().train.scheduler['class_meta'],
            'lr_scheduler_interval': HParams().train.scheduler['interval'],
            'max_norm_value_for_gradient_clip': getattr(HParams().train,'max_norm_value_for_gradient_clip',None),
            # train paremeters
            'total_step': getattr(HParams().train, 'total_step', np.inf),
            'total_epoch': getattr(HParams().train, 'total_epoch', int(1e20)),
            'seed': (int)(torch.cuda.initial_seed() / (2**32)) if HParams().train.seed is None else HParams().train.seed,
            'seed_strict': HParams().train.seed_strict,
            # logging
            'logger': logger,
            'save_model_step_interval': getattr(HParams().train, 'save_model_step_interval', None),
            'save_model_epoch_interval': getattr(HParams().train, 'save_model_epoch_interval', 1),
            'log_step_interval': getattr(HParams().log, 'log_step_interval', 1),
            'start_logging_epoch': getattr(HParams().log, 'start_logging_epoch', 0),
            # resource
            'device': HParams().resource.device,
            'multi_gpu': getattr(HParams().resource, 'multi_gpu', False),
            # additional
            'check_evalstep_first': getattr(HParams().train,'check_evalstep_first',False),
            'debug_mode': getattr(HParams().mode, 'debug_mode', False),
            'use_torch_compile': getattr(HParams().mode, 'use_torch_compile', True),
        }

        train_class_meta:dict = HParams().train.class_meta # {'name': 'Trainer', 'args': {}}
        trainer_class_name:str = train_class_meta['name']
        trainer_args.update(train_class_meta['args'])
        
        trainer_class:Type[Trainer] = GetModule.get_module_class(
            class_type = 'trainer', 
            module_name = trainer_class_name
        )
        trainer:Trainer = trainer_class(**trainer_args)
        trainer.init_train()
        
        if HParams().mode.is_train_resume:
            Util.print('load the checkpoint', 'info')
            trainer.load_train(HParams().mode.train_resume_path + "/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self) -> None:
        from torch_jaekwon.inference.inferencer import Inferencer
        
        infer_class_meta:dict = HParams().inference.class_meta # {'name': 'Inferencer', 'args': {}}
        inferencer_args:dict = {
            'output_dir': ARTIFACTS_DIRS.inference,
            'model':  None,
            'model_class_meta': HParams().model.class_meta,
            'set_type': HParams().inference.set_type,
            'set_meta_dict': {
                'single': HParams().inference.testdata_path,
                'dir': HParams().inference.testdata_dir_path
            },
            'device': HParams().resource.device
        }
        inferencer_args.update(infer_class_meta['args'])
        if 'save_dir_name' not in inferencer_args: inferencer_args['save_dir_name'] =  HParams().mode.config_name

        inferencer_class:Type[Inferencer] = GetModule.get_module_class(
            class_type = "inferencer", 
            module_name = infer_class_meta['name']
        )
        inferencer:Inferencer = inferencer_class(**inferencer_args)
        inferencer.inference(
            pretrained_root_dir = HParams().inference.pretrain_root_dir,
            pretrained_dir_name = HParams().mode.config_name if HParams().inference.pretrain_dir == '' else HParams().inference.pretrain_dir,
            ckpt_name = HParams().inference.ckpt_name
        )

    def evaluate(self) -> None:
        from torch_jaekwon.evaluate.evaluator.evaluator import Evaluator
        evaluater_class:Type[Evaluator] = GetModule.get_module_class(
            class_type='evaluator', 
            module_name=self.eval_class_meta['name']
        )
        evaluater_args:dict = self.eval_class_meta['args']
        evaluater_args.update({
            'device': HParams().resource.device,
            'evaluation_result_dir': f'{ARTIFACTS_DIRS.evaluate}/{HParams().mode.config_name}'
        })
        evaluater:Evaluator = evaluater_class(**evaluater_args)
        evaluater.evaluate()
    
    def set_argparse(self) -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '-r',
            '--resume',
            help='train resume',
            action='store_true'
        )

        primitive_types = (str, int, float, bool)
        str2bool = lambda v: v if isinstance(v, bool) else True if v.lower() in ('yes', 'true', 't', '1') else False if v.lower() in ('no', 'false', 'f', '0') else (_ for _ in ()).throw(ValueError("Boolean value expected."))
        type_mapper = lambda t: str2bool if t == bool else t

        arg_list_from_h_params:list = [
            {
                'arg_name': attr_name, 
                'module_name': module_name, 
                'attr_name':attr_name, 
                'type': type_mapper(type(value[0]) if isinstance(value, list) else type(value)),
                'nargs': '*' if isinstance(value, list) else None
            }
            for module_name, instance in HParams().__dict__.items() 
            for attr_name, value in instance.__dict__.items() 
            if isinstance(value, primitive_types) or isinstance(value, list) #and all(isinstance(item, primitive_types) for item in value)
        ]

        arg_name_list_from_h_params = [h_prams_arg['arg_name'] for h_prams_arg in arg_list_from_h_params]
        assert len(arg_name_list_from_h_params) == len(set(arg_name_list_from_h_params)), "Duplicate argument names found in HParams."

        for h_prams_arg in arg_list_from_h_params:
            parser.add_argument(
                *[f'--{h_prams_arg["arg_name"]}'],
                nargs= h_prams_arg['nargs'],
                type=h_prams_arg['type'],
                required=False,
                default=None,
                help="",
            )       

        args = parser.parse_args()
        
        for h_prams_arg in arg_list_from_h_params:
            value = getattr(args, h_prams_arg['arg_name'])
            if value is not None:
                setattr(getattr(HParams(), h_prams_arg['module_name']), h_prams_arg['attr_name'], value)
                
        if args.config_path is not None: HParams().set_config(args.config_path)
        if args.resume: HParams().mode.is_train_resume = True

        return args
