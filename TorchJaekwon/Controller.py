from typing import Type
import argparse

from HParams import HParams
from TorchJaekwon.GetModule import GetModule

class Controller():
    def __init__(self) -> None:
        self.h_params = HParams()

    def run(self) -> None:
        print("=============================================")
        print(f"{self.h_params.mode.stage} start.")
        print("=============================================")
        config_name:str = self.h_params.mode.config_path.split("/")[-1]
        print(f"{config_name} start.")
        print("=============================================")
        
        getattr(self,self.h_params.mode.stage)()
        
        print("Finish app.")

    def preprocess(self) -> None:
        from TorchJaekwon.DataProcess.Preprocess.Preprocessor import Preprocessor
        for data_name in self.h_params.data.config_per_dataset_dict:
            for preprocessor_meta in self.h_params.data.config_per_dataset_dict[data_name]['preprocessor_class_meta_list']:
                preprocessor_class_name:str = preprocessor_meta['name']
                preprocessor_args:dict = {'data_name': data_name}
                preprocessor_args.update(preprocessor_meta['args'])

                preprocessor_class:Type[Preprocessor] = GetModule.get_module_class( "./DataProcess/Preprocess", preprocessor_class_name )
                preprocessor:Preprocessor = preprocessor_class(**preprocessor_args)                             
                preprocessor.preprocess_data()                           

    def train(self) -> None:
        from TorchJaekwon.Train.Trainer.Trainer import Trainer
        trainer_args:dict = self.h_params.train.class_meta['args']
        trainer_class:Type[Trainer] = GetModule.get_module_class('./Train/Trainer',self.h_params.train.class_meta['name'])
        trainer = trainer_class(**trainer_args)
        trainer.init_train()
        
        if self.h_params.mode.train == "resume":
            trainer.load_train(self.h_params.mode.resume_path+"/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self):
        from TorchJaekwon.Inference.Inferencer.Inferencer import Inferencer
        inferencer_class:Type[Inferencer] = GetModule.get_module_class("./Inference/Inferencer", self.h_params.inference.class_meta['name'])
        inferencer = inferencer_class(**self.h_params.inference.class_meta['args'])
        inferencer.inference()

    def evaluate(self) -> None:
        from TorchJaekwon.Evaluater.Evaluater import Evaluater
        evaluater_class:Type[Evaluater] = GetModule.get_module_class("./Evaluater", self.h_params.evaluate.class_meta['name'])
        evaluater:Evaluater = evaluater_class(**self.h_params.evaluate.class_meta['args'])
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
        
        if args.config_path is not None: self.h_params.set_config(args.config_path)
        if args.stage is not None: self.h_params.mode.stage = args.stage
        if args.log_visualizer is not None: self.h_params.log.visualizer_type = args.log_visualizer
        if args.resume: self.h_params.mode.train = "resume"
        

        return args