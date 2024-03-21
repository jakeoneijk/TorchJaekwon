from typing import Type, Literal, Dict
import argparse

from HParams import HParams
from TorchJaekwon.GetModule import GetModule

class Controller():
    def __init__(self) -> None:
        self.set_argparse()

        h_params = HParams()
        self.config_name:str = h_params.mode.config_name
        self.stage: Literal['preprocess', 'train', 'inference', 'evaluate'] = h_params.mode.stage

        self.config_per_dataset_dict: Dict[str, dict] = h_params.data.config_per_dataset_dict
        self.train_class_meta:dict = h_params.train.class_meta # {'name': 'Trainer', 'args': {}}
        self.train_mode: Literal['start', 'resume'] = h_params.mode.train
        self.train_resume_path: str = h_params.mode.resume_path
        self.infer_class_meta:dict = h_params.inference.class_meta # {'name': 'Inferencer', 'args': {}}
        self.eval_class_meta:dict = h_params.evaluate.class_meta # {'name': 'Evaluater', 'args': {}}

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
                preprocessor_args:dict = {'data_name': data_name}
                preprocessor_args.update(preprocessor_meta['args'])

                preprocessor_class:Type[Preprocessor] = GetModule.get_module_class( "./DataProcess/Preprocess", preprocessor_class_name )
                preprocessor:Preprocessor = preprocessor_class(**preprocessor_args)                             
                preprocessor.preprocess_data()                           

    def train(self) -> None:
        from TorchJaekwon.Train.Trainer.Trainer import Trainer
        trainer_args:dict = self.train_class_meta['args']
        trainer_class:Type[Trainer] = GetModule.get_module_class('./Train/Trainer', self.train_class_meta['name'])
        trainer:Trainer = trainer_class(**trainer_args)
        trainer.init_train()
        
        if self.train_mode == "resume":
            trainer.load_train(self.train_resume_path + "/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self) -> None:
        from TorchJaekwon.Inference.Inferencer.Inferencer import Inferencer
        inferencer_class:Type[Inferencer] = GetModule.get_module_class("./Inference/Inferencer", self.infer_class_meta['name'])
        inferencer:Inferencer = inferencer_class(**self.infer_class_meta['args'])
        inferencer.inference()

    def evaluate(self) -> None:
        from TorchJaekwon.Evaluater.Evaluater import Evaluater
        evaluater_class:Type[Evaluater] = GetModule.get_module_class("./Evaluater", self.eval_class_meta['name'])
        evaluater:Evaluater = evaluater_class(**self.eval_class_meta['args'])
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
            "-lv",
            "--log_visualizer",
            type=str,
            required=False,
            default=None,
            choices = ['tensorboard', 'wandb'],
            help="",
        )

        args = parser.parse_args()

        h_params = HParams()
        if args.config_path is not None: h_params.set_config(args.config_path)
        if args.stage is not None: h_params.mode.stage = args.stage
        if args.log_visualizer is not None: h_params.log.visualizer_type = args.log_visualizer
        if args.resume: h_params.mode.train = "resume"
        

        return args