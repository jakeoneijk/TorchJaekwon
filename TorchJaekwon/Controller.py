from typing import Type, Literal, Dict, List, Union, Literal
import argparse

from HParams import HParams
from TorchJaekwon.GetModule import GetModule

class Controller():
    def __init__(self, 
                 additional_args: List[Dict[Literal['args','kwargs','replace_var'], Union[list, dict, object]]] = list()
                 ) -> None:
        self.set_argparse(additional_args)

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
        from TorchJaekwon.DataProcess.Preprocess.Preprocessor import Preprocessor
        for data_name in self.config_per_dataset_dict:
            for preprocessor_meta in self.config_per_dataset_dict[data_name]['preprocessor_class_meta_list']:
                preprocessor_class_name:str = preprocessor_meta['name']
                preprocessor_args:dict = {
                    'data_name': data_name,
                    'root_dir': HParams().data.root_path,
                    'num_workers': HParams().resource.preprocess['num_workers']
                }
                preprocessor_args.update(preprocessor_meta['args'])

                preprocessor_class:Type[Preprocessor] = GetModule.get_module_class( "./DataProcess/Preprocess", preprocessor_class_name )
                preprocessor:Preprocessor = preprocessor_class(**preprocessor_args)                             
                preprocessor.preprocess_data()                           

    def train(self) -> None:
        import torch
        from TorchJaekwon.Train.Trainer.Trainer import Trainer
        
        train_class_meta:dict = HParams().train.class_meta # {'name': 'Trainer', 'args': {}}
        trainer_args:dict = {
            'device': HParams().resource.device,
            'model_class_name': HParams().model.class_name,
            'model_class_meta_dict': HParams().model.class_meta_dict,
            'loss_class_meta': HParams().train.loss_dict,
            'max_norm_value_for_gradient_clip': getattr(HParams().train,'max_norm_value_for_gradient_clip',None),
            'total_epoch': HParams().train.epoch,
            'save_model_every_step': getattr(HParams().train, 'save_model_every_step', None),
            'seed': (int)(torch.cuda.initial_seed() / (2**32)) if HParams().train.seed is None else HParams().train.seed,
            'seed_strict': HParams().train.seed_strict
        }
        trainer_args.update(train_class_meta['args'])
        
        trainer_class:Type[Trainer] = GetModule.get_module_class('./Train/Trainer', train_class_meta['name'])
        trainer:Trainer = trainer_class(**trainer_args)
        trainer.init_train()
        
        if self.train_mode == "resume":
            trainer.load_train(self.train_resume_path + "/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self) -> None:
        from TorchJaekwon.Inference.Inferencer.Inferencer import Inferencer
        
        infer_class_meta:dict = HParams().inference.class_meta # {'name': 'Inferencer', 'args': {}}
        inferencer_args:dict = {
            'output_dir': HParams().inference.output_dir,
            'experiment_name': HParams().mode.config_name,
            'model':  None,
            'model_class_name': HParams().model.class_name,
            'device': HParams().resource.device
        }
        inferencer_args.update(infer_class_meta['args'])

        inferencer_class:Type[Inferencer] = GetModule.get_module_class("./Inference/Inferencer", infer_class_meta['name'])
        inferencer:Inferencer = inferencer_class(**inferencer_args)
        inferencer.inference(
            pretrained_root_dir = HParams().inference.pretrain_root_dir,
            pretrained_dir_name = HParams().mode.config_name if HParams().inference.pretrain_dir == '' else HParams().inference.pretrain_dir,
            pretrain_module_name = HParams().inference.pretrain_module_name
        )

    def evaluate(self) -> None:
        from TorchJaekwon.Evaluater.Evaluater import Evaluater
        evaluater_class:Type[Evaluater] = GetModule.get_module_class("./Evaluater", self.eval_class_meta['name'])
        evaluater:Evaluater = evaluater_class(**self.eval_class_meta['args'])
        evaluater.evaluate()
    
    def set_argparse(self,
                     additional_args:List[Dict[Literal['args','kwargs','replace_var'], Union[list, dict, object]]] = list()
                     ) -> None:
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

        return args