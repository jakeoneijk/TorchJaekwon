import os
from abc import ABC,abstractmethod
import numpy as np
import yaml

from HParams import HParams

class Evaluater(ABC):
    def __init__(self, h_params:HParams):
        self.h_params = h_params
        self.output_dir = self.h_params.test.output_path
        self.evaluation_pretrained_name = self.h_params.evaluate.source_dir_name
        self.data_path = f"{self.output_dir}/{self.evaluation_pretrained_name}"
        self.save_name:str = f"{self.output_dir}/{self.h_params.mode.config_name}_evaluation.yaml"
    
    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''
    @abstractmethod
    def read_pred_gt_list(self,data_name):
        '''
        return {key1:{"reference":data,"estimate":data},key2...}
        '''
        pass

    @abstractmethod
    def evaluator(self,test_set_dict):
        '''
        return evaluation resutl
        '''
        pass
    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''

    def process_multiple(self):
        root_path = self.data_path
        data_dir_path_list = os.listdir(self.data_path)

        for data_dir in data_dir_path_list:
            self.data_path = root_path + "/" + data_dir
            self.process()

    
    def process(self):
        evaluation_results_list_dict = dict()
        data_path_list = os.listdir(self.data_path)
        data_path_list.sort()
        for i,data_name in enumerate(data_path_list):
            print(f"{i+1}/{len(data_path_list)}")
            test_set_dict:dict = self.read_pred_gt_list(data_name)
            evaluation_result = self.evaluator(test_set_dict)
            evaluation_results_list_dict = self.append_result(evaluation_results_list_dict,evaluation_result)
        feature_of_results = self.extract_feature_from_results(evaluation_results_list_dict)
        self.report_and_save_result(feature_of_results)
    

    def append_result(self,results_dict, new_result_dict):
        for data_name in new_result_dict:
            if data_name not in results_dict:
                results_dict[data_name] = dict()
            for metric_name in new_result_dict[data_name]:
                if metric_name not in results_dict[data_name]:
                    results_dict[data_name][metric_name] = []
                results_dict[data_name][metric_name].append(new_result_dict[data_name][metric_name])
        return results_dict

    def extract_feature_from_results(self,results_dict):
        '''
        median max min for each feature
        '''
        median_feature_dict = dict()
        for data_name in results_dict:
            median_feature_dict[data_name] = dict()
            for metric_name in results_dict[data_name]:
                median_feature_dict[data_name][f"{metric_name}_median"]= np.median(results_dict[data_name][metric_name])
                median_feature_dict[data_name][f"{metric_name}_mean"]= np.mean(results_dict[data_name][metric_name])
        return median_feature_dict

    def report_and_save_result(self,result_dict):
        '''
        print and yaml save
        '''
        for data_name in result_dict:
            print("================================================")
            print(data_name)
            print("================================================")
            for metric_name in result_dict[data_name]:
                if "numpy" in str(type(result_dict[data_name][metric_name])):
                    result_dict[data_name][metric_name] = result_dict[data_name][metric_name].item()
                print(f"{metric_name} : {result_dict[data_name][metric_name]}")
        
        with open(self.save_name,'w') as file:
            yaml.dump(result_dict, file)
        

    

