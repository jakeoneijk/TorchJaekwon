import os

class InitProject:
    def __init__(self) -> None:
        self.project_dir:str = "."
        self.copy_dir:str = "./TorchJAEKWON/cp"
        self.dir_list:list =[
            "Data/Dataset",
            "Data/PytorchDataLoader/BatchSampler",
            "Data/PytorchDataset",

            "DataProcess/MakeMetaData",
            "DataProcess/Preprocess",
            "DataProcess/Util",

            "Train/LogWriter",
            "Train/Loss/LossControl",
            "Train/Loss/LossFunction",
            "Train/Optimizer",
            "Train/Trainer",

            "Inference/Output",
            "Inference/Inferencer",

            "Evaluater",
        ]
    
    def init_project(self):
        os.system("pwd")
        os.system(f"cp -r {self.copy_dir}/* {self.project_dir}")
        for dir in self.dir_list:
            os.makedirs(f"{self.project_dir}/{dir}",exist_ok=True)

if __name__ == "__main__":
    init_project = InitProject()
    init_project.init_project()