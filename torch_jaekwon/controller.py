#type
from typing import Type, Literal

#package
import os
import sys
import argparse
import numpy as np
import torch

#torchjaekwon
from .h_params import HParams
from . import get_module
from .get_module import GetModule
from .util import util, util_data
from . import path as tj_path

def run() -> None:
    config_dict:dict = set_argparse()
    config_name:str = config_dict['cli']['config_name']
    stage: Literal['preprocess', 'train', 'inference', 'evaluate'] = config_dict['cli']['stage']

    util.log(f"[{stage}] {config_name} start.", msg_type='info')
    getattr(sys.modules[__name__],stage)(config_dict)
    util.log(f"[{stage}] {config_name} finish.", msg_type='success')

def set_argparse() -> dict:
    parser = argparse.ArgumentParser()
    str2bool = lambda v: v if isinstance(v, bool) else True if v.lower() in ('yes', 'true', 't', '1') else False if v.lower() in ('no', 'false', 'f', '0') else (_ for _ in ()).throw(ValueError("Boolean value expected."))

    # common arguments
    parser.add_argument('--config_path', type=str, help='config file path')
    parser.add_argument('--config_name', type=str, help='config name for logging', required=False)
    parser.add_argument('--project_name', type=str, help='project name for logging')
    parser.add_argument('--stage', type=str, help='stage: preprocess | train | inference | evaluate')

    # resource arguments
    parser.add_argument('--num_workers', type=int, help='number of workers for data loading', default=1)

    # train arguments
    parser.add_argument('-r', '--resume', help='train resume', action='store_true')
    parser.add_argument('--train_resume_path', type=str, help='train resume path', required=False)
    parser.add_argument('--check_evalstep_first', type=str2bool, help='check evalstep first', default=True)
    parser.add_argument('--debug_mode', type=str2bool, help='debug mode', default=True)
    parser.add_argument('--use_torch_compile', type=str2bool, help='use torch compile', default=True)
    parser.add_argument('--log_tool', type=str, help='log tool: tensorboard | wandb', default='tensorboard')
    parser.add_argument('--log_step_interval', type=int, help='log step interval', default=40)
    parser.add_argument('--start_logging_epoch', type=int, help='start logging epoch', default=0)
    parser.add_argument('--save_model_epoch_interval', type=int, help='save model epoch interval', default=100)

    # inference arguments
    parser.add_argument('--infer_data_path', nargs='*', type=str, help='inference data path when set type is single or dir', required=False)
    parser.add_argument('--ckpt_name', type=str, help='checkpoint name for inference', default='last')

    # evaluate arguments
    parser.add_argument('--eval_gt_dir_path', type=str, help='evaluation directory path for groundtruth', required=False)
    parser.add_argument('--eval_pred_dir_path', type=str, help='evaluation directory path for prediction', required=False)

    args = parser.parse_args()

    config_dict = util_data.yaml_load(args.config_path)
    assert 'cli' not in config_dict, "Reserved key 'cli' found in config file."
    config_dict['cli'] = vars(args)
    config_dict['cli']['config_name'] = config_dict['cli']['config_name'] or os.path.splitext(tj_path.relpath(args.config_path, start_dir_path=tj_path.CONFIG_DIR))[0]
    config_dict['cli']['train_resume_path'] = config_dict['cli'].get('train_resume_path', f"{tj_path.ARTIFACTS_DIRS.train}/{config_dict['cli']['config_name']}")

    ## Legacy #########################
    primitive_types = (str, int, float, bool)
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
    arg_name_list_from_h_params.sort()
    assert len(arg_name_list_from_h_params) == len(set(arg_name_list_from_h_params)), "Duplicate argument names found in HParams."

    if args.config_path is not None: HParams().set_config(args.config_path)
    
    for h_prams_arg in arg_list_from_h_params:
        value = getattr(args, h_prams_arg['arg_name'], None)
        if value is not None:
            setattr(getattr(HParams(), h_prams_arg['module_name']), h_prams_arg['attr_name'], value)
    ## Legacy #########################
    return config_dict

def preprocess(config_dict:dict) -> None:
    from .data.preprocess.preprocessor import Preprocessor
    preprocessor_class_meta_list:list = HParams().data.preprocessor_class_meta_list
    num_workers:int = HParams().resource.num_workers
    device:torch.device = HParams().resource.device
    
    for preprocessor_meta in preprocessor_class_meta_list:
        preprocessor_meta['args']['num_workers'] = preprocessor_meta['args'].get('num_workers', num_workers)
        preprocessor_meta['args']['device'] = preprocessor_meta['args'].get('device', device)
        preprocessor:Preprocessor = get_module.get_module_tj(class_type='preprocessor', class_meta=preprocessor_meta)
        preprocessor.preprocess_data()                           

def train(config_dict:dict) -> None:
    import torch
    from torch_jaekwon.train.trainer.trainer import Trainer
    from torch_jaekwon.train.logger.logger import Logger

    logger = Logger(
        experiment_name = config_dict['cli']['config_name'],
        use_time_on_experiment_name = False,
        project_name = config_dict['cli'].get('project_name') or 'default_project',
        visualizer_type = config_dict['cli']['log_tool'],
        root_dir_path = f'{tj_path.ARTIFACTS_DIRS.train}/{config_dict["cli"]["config_name"]}',
        is_resume = config_dict['cli']['resume'],
    )
    config_dict['dataloader']['train']['dataset_class_meta']['args']['logger'] = logger

    trainer_args = {
        # data
        'data_class_meta_dict': config_dict['dataloader'],
        # model
        'model_class_meta_dict': HParams().model.class_meta,
        # loss
        'loss_meta_dict': getattr(HParams().train, 'loss', None),
        # optimizer
        'optimizer_class_meta_dict': HParams().train.optimizer['class_meta'],
        'lr_scheduler_class_meta_dict': HParams().train.scheduler['class_meta'],
        'lr_scheduler_interval': HParams().train.scheduler['interval'],
        # train paremeters
        'seed': (int)(torch.cuda.initial_seed() / (2**32)) if HParams().train.seed is None else HParams().train.seed,
        'seed_strict': HParams().train.seed_strict,
        # logging
        'logger': logger,
        'save_model_step_interval': getattr(HParams().train, 'save_model_step_interval', None),
        'save_model_epoch_interval': getattr(HParams().train, 'save_model_epoch_interval', 1),
        'log_step_interval': getattr(HParams().log, 'log_step_interval', 1),
        'start_logging_epoch': getattr(HParams().log, 'start_logging_epoch', 0),
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
    
    if config_dict['cli']['resume']:
        util.log('load the checkpoint', 'info')
        trainer.load_train(HParams().mode.train_resume_path + "/train_checkpoint.pth")
    
    trainer.fit()

def inference(config_dict:dict) -> None:
    from torch_jaekwon.inference.inferencer import Inferencer
    
    infer_class_meta:dict = config_dict['inference']['class_meta']
    inferencer_args:dict = {
        'output_dir': tj_path.ARTIFACTS_DIRS.inference,
        'model':  None,
        'model_class_meta': config_dict['model']['class_meta'],
        'input_data_path': config_dict['cli']['infer_data_path'],
    }
    inferencer_args.update(infer_class_meta['args'])
    if 'save_dir_name' not in inferencer_args: inferencer_args['save_dir_name'] =  config_dict['cli']['config_name']

    inferencer_class:Type[Inferencer] = GetModule.get_module_class(
        class_type = "inferencer", 
        module_name = infer_class_meta['name']
    )
    inferencer:Inferencer = inferencer_class(**inferencer_args)
    inferencer.inference(
        pretrained_root_dir = tj_path.ARTIFACTS_DIRS.train,
        pretrained_dir_name = config_dict['cli']['config_name'],
        ckpt_name = config_dict['cli']['ckpt_name']
    )

def evaluate(config_dict:dict) -> None:
    from torch_jaekwon.evaluate.evaluator.evaluator import Evaluator
    eval_class_meta:dict = config_dict['evaluate']['class_meta']
    config_name:str = config_dict['cli']['config_name']
    gt_dir_path:str = config_dict['cli']['eval_gt_dir_path']
    pred_dir_path:str = config_dict['cli']['eval_pred_dir_path']

    evaluater_class:Type[Evaluator] = GetModule.get_module_class(class_type='evaluator', module_name=eval_class_meta['name'])
    evaluater_args:dict = eval_class_meta['args']
    evaluater_args.update({'pred_dir_path': pred_dir_path, 'gt_dir_path': gt_dir_path, 'result_dir_path': f'{tj_path.ARTIFACTS_DIRS.evaluate}/{config_name}'})
    evaluater:Evaluator = evaluater_class(**evaluater_args)
    evaluater.evaluate()
