from concurrent.futures import ProcessPoolExecutor
import os
import time
import torch
from tqdm import tqdm
from ...path import ARTIFACTS_DIRS

class Preprocessor():
    def __init__(
        self,
        data_name:str = None,
        root_dir:str = ARTIFACTS_DIRS.preprocessed_data,
        device:torch.device = None,
        num_workers:int = 1,
    ) -> None:
        # args to class variable
        self.data_name:str = data_name
        self.root_dir:str = root_dir
        self.num_workers:int = num_workers
        self.device:torch.device = device
        self.max_meta_data_len:int = 10000
        if self.root_dir is not None and self.data_name is not None:
            self.output_dir = self.get_output_dir()
        else:
            print('Warning: root_dir or data_name is None')
    
    # ==========================
    # Methods to Override (Start)
    # ==========================
    
    def get_meta_data_param(self) -> list:
        '''
        meta_data_param_list = list()
        '''
        raise NotImplementedError
    
    def preprocess_one_data(self, param:dict) -> None:
        '''
        ex) (subset, file_name) = param
        '''
        raise NotImplementedError
    
    def final_process(self, result_list:list) -> None:
        print("Finish preprocess")
    
    # ==========================
    # Methods to Override (End)
    # ==========================
    
    def get_output_dir(self) -> str:
        return os.path.join(self.root_dir, self.data_name)
    
    def write_message(self, message_type:str, message:str) -> None:
        with open(f"{self.preprocessed_data_path}/{message_type}.txt",'a') as file_writer:
            file_writer.write(message+'\n')
    
    def preprocess_data(self) -> None:
        meta_param_list:list = self.get_meta_data_param()
        print(f'length of meta data: {len(meta_param_list)}')
        result_list = list()
        start_time:float = time.time()
        for start_idx in tqdm(range(0,len(meta_param_list),self.max_meta_data_len),desc='sub meta param list'):
            sub_meta_param_list = meta_param_list[start_idx:start_idx+self.max_meta_data_len]
            if sub_meta_param_list is None:
                print('meta_param_list is None, So we skip preprocess data')
                return
            if self.num_workers > 2:
                with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                    # pool.map(self.preprocess_one_data, sub_meta_param_list)
                    with tqdm(total=len(sub_meta_param_list)) as progress:
                        future_list = list()

                        for sub_meta_param in sub_meta_param_list:
                            future = pool.submit(self.preprocess_one_data, sub_meta_param)
                            future.add_done_callback(lambda p: progress.update())
                            future_list.append(future)

                        for future in future_list:
                            result = future.result()
                            if result is not None: result_list.append(result)
            else:
                for meta_param in tqdm(sub_meta_param_list,desc='preprocess data'):
                    result = self.preprocess_one_data(meta_param)
                    if result is not None: result_list.append(result)

        self.final_process(result_list)
        print("{:.3f} s".format(time.time() - start_time))