from datetime import timedelta
import os
from tqdm import tqdm
import torch
import torch.distributed as distributed
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from ...util import util, util_torch_distributed
from ...path import ARTIFACTS_DIRS

class TorchrunPreprocessor():
    def __init__(
        self, 
        data_name:str = None,
        root_dir:str = ARTIFACTS_DIRS.preprocessed_data,
        batch_size:int = 1,
        num_workers:int = 0,
        **kwargs
    ) -> None:
        self.data_name:str = data_name
        self.root_dir:str = root_dir
        if self.root_dir is not None and self.data_name is not None:
            self.output_dir = self.get_output_dir()
            os.makedirs(self.output_dir,exist_ok=True)
        
        self.batch_size:int = batch_size
        self.num_workers:int = num_workers

        distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        torch.cuda.set_device(util_torch_distributed.local_rank())

    def get_output_dir(self) -> str:
        return os.path.join(self.root_dir, self.data_name)
    
    def get_dataset(self) -> Dataset:
        '''
        Returns the dataset for the preprocessor.
        This method should be overridden in subclasses.
        '''
        raise NotImplementedError("Subclasses must implement this method.")

    def preprocess_batch(self, batch:dict) -> None:
        '''
        Preprocess a batch of data.
        This method should be overridden in subclasses.
        '''
        raise NotImplementedError("Subclasses must implement this method.")

    def final_process(self) -> None:
        util.log("Finish preprocess", msg_type='success')

    def preprocess_data(self) -> None:
        dataset:Dataset = self.get_dataset()
        dataloader:DataLoader = util_torch_distributed.get_dataloader(
            dataloader_args={
                'dataset': dataset,
                'drop_last': False,
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
            },
            shuffle=False
        )
        util.log(f'Number of samples: {len(dataset)}', msg_type='info')
        util.log(f'Number of batches: {len(dataloader)}', msg_type='info')
        for data in tqdm(dataloader):
            self.preprocess_batch(data)
        distributed.barrier()
        if util_torch_distributed.is_main_process():
            self.final_process()
        distributed.destroy_process_group()

