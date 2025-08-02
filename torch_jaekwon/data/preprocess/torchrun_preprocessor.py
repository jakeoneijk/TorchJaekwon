from datetime import timedelta
import os
import torch
import torch.distributed as distributed
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ...util import util_torch_distributed
from ...path import ARTIFACTS_DIRS

class TorchrunPreprocessor():
    def __init__(
        self, 
        data_name:str = None,
        root_dir:str = ARTIFACTS_DIRS.preprocessed_data,
        **kwargs
    ) -> None:
        self.data_name:str = data_name
        self.root_dir:str = root_dir
        if self.root_dir is not None and self.data_name is not None:
            self.output_dir = self.get_output_dir()
            os.makedirs(self.output_dir,exist_ok=True)

        distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        distributed_config:dict = util_torch_distributed.get_config()
        torch.cuda.set_device(distributed_config['local_rank'])

    def get_output_dir(self) -> str:
        return os.path.join(self.root_dir, self.data_name)
    
    def get_dataset(self) -> Dataset:
        '''
        Returns the dataset for the preprocessor.
        This method should be overridden in subclasses.
        '''
        raise NotImplementedError("Subclasses must implement this method.")

    def preprocess_data(self) -> None:
        dataset:Dataset = self.get_dataset()
