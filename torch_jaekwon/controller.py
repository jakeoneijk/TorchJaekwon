#type
from typing import Type, Literal, Dict, List, Union, Literal

#package
import os
import argparse
import numpy as np

#torchjaekwon
from .get_module import GetModule
from .util import Util

#internal
from h_params import HParams

class Controller():
    def __init__(self) -> None:
        self.set_argparse()

        self.config_name:str = HParams().mode.config_name
        self.stage: Literal['preprocess', 'train', 'inference', 'evaluate'] = HParams().mode.stage

        self.config_per_dataset_dict: Dict[str, dict] = HParams().data.config_per_dataset_dict
        self.train_mode: Literal['start', 'resume'] = HParams().mode.train
        self.train_resume_path: str = HParams().mode.resume_path
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
        from .data_process.preprocess.preprocessor import Preprocessor
        for data_name in self.config_per_dataset_dict:
            for preprocessor_meta in self.config_per_dataset_dict[data_name]['preprocessor_class_meta_list']:
                preprocessor_class_name:str = preprocessor_meta['name']
                preprocessor_args:dict = {
                    'data_name': data_name,
                    'root_dir': HParams().data.root_path,
                    'num_workers': HParams().resource.preprocess['num_workers'],
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
        if self.train_mode == "resume": Util.print('resume the training', 'info')   
        import torch
        from torch_jaekwon.train.trainer.trainer import Trainer
        trainer_args = {
            # data
            'data_class_meta_dict': HParams().pytorch_data.class_meta,
            # model
            'model_class_name': HParams().model.class_name,
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
        
        if self.train_mode == "resume":
            Util.print('load the checkpoint', 'info')
            trainer.load_train(self.train_resume_path + "/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self) -> None:
        from torch_jaekwon.Inference.Inferencer.Inferencer import Inferencer
        
        infer_class_meta:dict = HParams().inference.class_meta # {'name': 'Inferencer', 'args': {}}
        inferencer_args:dict = {
            'output_dir': HParams().inference.output_dir,
            'experiment_name': HParams().mode.config_name,
            'model':  None,
            'model_class_name': HParams().model.class_name,
            'set_type': HParams().inference.set_type,
            'set_meta_dict': HParams().inference.set_meta_dict,
            'device': HParams().resource.device
        }
        inferencer_args.update(infer_class_meta['args'])

        inferencer_class:Type[Inferencer] = GetModule.get_module_class(
            root_path = "./Inference/Inferencer", 
            module_name = infer_class_meta['name']
        )
        inferencer:Inferencer = inferencer_class(**inferencer_args)
        inferencer.inference(
            pretrained_root_dir = HParams().inference.pretrain_root_dir,
            pretrained_dir_name = HParams().mode.config_name if HParams().inference.pretrain_dir == '' else HParams().inference.pretrain_dir,
            ckpt_name = HParams().inference.ckpt_name
        )

    def evaluate(self) -> None:
        from torch_jaekwon.Evaluater.Evaluater import Evaluater
        evaluater_class:Type[Evaluater] = GetModule.get_module_class(
            root_path="./Evaluater", 
            module_name=self.eval_class_meta['name']
        )
        evaluater_args:dict = self.eval_class_meta['args']
        evaluater_args.update({
            'device': HParams().resource.device
        })
        if evaluater_args.get('source_dir','') == '':
            source_dir_prefix:str = f'{HParams().inference.output_dir}/{HParams().mode.config_name}'
            source_dir_parent:str = '/'.join(source_dir_prefix.split('/')[:-1])
            source_dir_tag:str = source_dir_prefix.split('/')[-1]
            source_dir_name_candidate = [dir_name for dir_name in os.listdir(source_dir_parent) if source_dir_tag in dir_name]
            source_dir_name_candidate.sort()
            evaluater_args['source_dir'] = f'{source_dir_parent}/{source_dir_name_candidate[-1]}'
        evaluater:Evaluater = evaluater_class(**evaluater_args)
        evaluater.evaluate()
    
    def set_argparse(self) -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-c",
            "--config_path",
            type=str,
            required=False,
            default=None,
            help="",
        )

        parser.add_argument(
            "-s",
            "--stage",
            type=str,
            required=False,
            default=None,
            choices = ['preprocess', 'train', 'inference', 'evaluate'],
            help="",
        )

        parser.add_argument(
            '-r',
            '--resume',
            help='train resume',
            action='store_true'
        )

        parser.add_argument(
            "-do",
            "--debug_off",
            help="debug mode off",
            action='store_true'
        )

        parser.add_argument(
            "-lv",
            "--log_visualizer",
            type=str,
            required=False,
            default=None,
            choices = ['tensorboard', 'wandb'],
            help="",
        )

        args = parser.parse_args()

        if args.config_path is not None: HParams().set_config(args.config_path)
        if args.stage is not None: HParams().mode.stage = args.stage
        if args.log_visualizer is not None: HParams().log.visualizer_type = args.log_visualizer
        if args.resume: HParams().mode.train = "resume"
        if args.debug_off: HParams().mode.debug_mode = False

        return args