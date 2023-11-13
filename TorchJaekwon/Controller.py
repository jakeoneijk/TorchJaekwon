import argparse

from HParams import HParams
from TorchJaekwon.GetModule import GetModule

class Controller():
    def __init__(self) -> None:
        self.h_params = HParams()
    
    def set_argparse(self) -> None:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "-s",
            "--stage",
            type=str,
            required=False,
            default=None,
            choices = ['preprocess', 'make_meta_data', 'train', 'inference', 'evaluate'],
            help="",
        )

        args = parser.parse_args()

        if args.stage is not None: self.h_params.mode.stage = args.stage

        return args

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
        for data_name in self.h_params.data.data_config_per_dataset_dict:
            preprocessor_args:dict = {'data_name': data_name, "data_config_dict": self.h_params.data.data_config_per_dataset_dict[data_name]}
            if 'preprocessor_args' in self.h_params.data.data_config_per_dataset_dict[data_name]:
                preprocessor_args.update(self.h_params.data.data_config_per_dataset_dict[data_name]['preprocessor_args'])
            preprocessor: Preprocessor = GetModule.get_module_class(
                "./DataProcess/Preprocess", 
                self.h_params.data.data_config_per_dataset_dict[data_name]["preprocessor_class_name"]
                )(**preprocessor_args)
                                                                    
            
            preprocessor.preprocess_data()
    
    def make_meta_data(self) -> None:
        from TorchJaekwon.DataProcess.MakeMetaData.MakeMetaData import MakeMetaData
        for mata_data_class_name in self.h_params.make_meta_data.process_dict:
            meta_data_maker: MakeMetaData = GetModule.get_module_class(root_path='./DataProcess/MakeMetaData',
                                                                       module_name=mata_data_class_name
                                                                       )(**self.h_params.make_meta_data.process_dict[mata_data_class_name])
            meta_data_maker.make_meta_data()

    def train(self) -> None:
        from TorchJaekwon.Train.Trainer.Trainer import Trainer
        trainer:Trainer = GetModule.get_module_class('./Train/Trainer',self.h_params.train.class_name)(**self.h_params.train.trainer_args)
        trainer.init_train()
        
        if self.h_params.mode.train == "resume":
            trainer.load_train(self.h_params.mode.resume_path+"/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self):
        from TorchJaekwon.Inference.Inferencer.Inferencer import Inferencer
        inferencer:Inferencer = GetModule.get_module_class("./Inference/Inferencer", self.h_params.inference.class_name)()
        inferencer.inference()

    def evaluate(self) -> None:
        from TorchJaekwon.Evaluater.Evaluater import Evaluater
        evaluater:Evaluater = GetModule.get_module_class("./Evaluater", self.h_params.evaluate.class_name)()
        evaluater.evaluate()