from HParams import HParams
from TorchJAEKWON.GetModule import GetModule
from TorchJAEKWON.DataProcess.Preprocess.Preprocessor import Preprocessor
from TorchJAEKWON.DataProcess.MakeMetaData.MakeMetaData import MakeMetaData
from TorchJAEKWON.Train.Trainer.Trainer import Trainer
from TorchJAEKWON.Inference.Inferencer.Inferencer import Inferencer
from TorchJAEKWON.Evaluater.Evaluater import Evaluater


class Controller():
    def __init__(self) -> None:
        self.h_params = HParams()

    def run(self) -> None:
        print("=============================================")
        print(f"{self.h_params.mode.app} start.")
        print("=============================================")
        config_name:str = self.h_params.mode.config_path.split("/")[-1]
        print(f"{config_name} start.")
        print("=============================================")
        
        if self.h_params.mode.app == "preprocess":
            self.preprocess()
        
        if self.h_params.mode.app == "make_meta_data":
            self.make_meta_data()

        if self.h_params.mode.app == "train":
            self.train()
        
        if self.h_params.mode.app == "inference":
            self.inference()

        if self.h_params.mode.app == "evaluate":
            self.evaluate()
        
        print("Finish app.")

    def preprocess(self) -> None:
        for data_name in self.h_params.data.data_config_per_dataset_dict:
            preprocessor: Preprocessor = GetModule.get_module_class(
                "./DataProcess/Preprocess", 
                self.h_params.data.data_config_per_dataset_dict[data_name]["preprocessor_class_name"]
                )(**{"data_config_dict": self.h_params.data.data_config_per_dataset_dict[data_name]})
                                                                    
            
            preprocessor.preprocess_data()
    
    def make_meta_data(self) -> None:
        for mata_data_class_name in self.h_params.make_meta_data.process_dict:
            meta_data_maker: MakeMetaData = self.get_module.get_module(module_type="make_meta_data", 
                                                                    module_name=mata_data_class_name,
                                                                    module_arg={"make_meta_data_config": self.h_params.make_meta_data.process_dict[mata_data_class_name]},
                                                                    arg_unpack=True)
            meta_data_maker.make_meta_data()

    def train(self) -> None:
        trainer:Trainer = GetModule.get_module('./Train/Trainer',self.h_params.train.class_name,None)
        trainer.init_train()
        
        if self.h_params.mode.train == "resume":
            trainer.load_train(self.h_params.mode.resume_path+"/train_checkpoint.pth")
        
        trainer.fit()

    def inference(self):
        inferencer:Inferencer = GetModule.get_module("./Inference/Inferencer", self.h_params.inference.class_name,arg_unpack=True)
        inferencer.inference()

    def evaluate(self):
        evaluater:Evaluater = self.get_module.get_module("evaluater", self.h_params.evaluate.class_name, module_arg=self.h_params,arg_unpack=False)
        evaluater.process()

def run_main(config_path=None) -> None:
    controller:Controller = Controller()

    if config_path is not None:
        h_params = HParams()
        h_params.set_config(config_path)

    controller.run()

if __name__ == '__main__':
    run_main()