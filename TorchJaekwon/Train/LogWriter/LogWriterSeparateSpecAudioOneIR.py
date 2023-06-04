import torch.nn as nn
from numpy import ndarray
from torch import Tensor

import soundfile as sf
import torch
import pickle
import numpy as np

import wandb
from Train.LogWriter.LogWriter import LogWriter
from DataProcess.Process.ProcessReverbOneIR import ProcessReverbOneIR

class LogWriterSeparateSpecAudioOneIR(LogWriter):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.test_data = self.get_test_data_dict()
        for feature_name in self.test_data:
            sf.write(f"{self.log_path['root']}/test_{feature_name}.wav",self.test_data[feature_name].T,samplerate=self.h_params.preprocess.sample_rate)
        
        
    def get_test_data_dict(self)->dict:
        data_path = "./Data/Dataset/musmain_musdb18_accom/test/Bobby Nobody - Stitch Up"
        vocal_file_name = "audio_vocal.pkl"
        accom_file_name = "audio_accompaniment.pkl"
        start_second = 66
        end_second = 72
        vocal = self.read_feature_pickle(f"{data_path}/{vocal_file_name}")[...,start_second*self.h_params.preprocess.sample_rate:end_second*self.h_params.preprocess.sample_rate]
        accom = self.read_feature_pickle(f"{data_path}/{accom_file_name}")[...,start_second*self.h_params.preprocess.sample_rate:end_second*self.h_params.preprocess.sample_rate]
        
        vocal_tensor:Tensor = torch.from_numpy(vocal).unsqueeze(0)
        reverb_processsor = ProcessReverbOneIR(self.h_params)
        reverb:Tensor = reverb_processsor.get_reverb(vocal_tensor)
        reverberated_audio_numpy:ndarray = reverb.squeeze().numpy()

        test_data = dict()
        test_data["target"] = vocal
        test_data["target with reverb"] = test_data["target"] + reverberated_audio_numpy
        test_data["input"] = test_data["target with reverb"] + accom
        return test_data

    def read_feature_pickle(self,data_path):
        with open(data_path, 'rb') as pickle_file:
            feature = pickle.load(pickle_file)
        return feature
    
    def log_every_epoch(self,model:nn.Module):
        if not self.h_params.resource.multi_gpu:
            with torch.no_grad():
                separated_vocal:dict = model(torch.from_numpy(self.test_data["input"]).unsqueeze(0).to(self.h_params.resource.device))
        else:
            with torch.no_grad():
                separated_vocal:dict = model.module(torch.from_numpy(self.test_data["input"]).unsqueeze(0).to(self.h_params.resource.device))
        for feature_name in separated_vocal:
            separated_vocal[feature_name] = separated_vocal[feature_name].to("cpu").squeeze().numpy()
        if self.h_params.log.visualizer_type == "wandb":
            wandb.log({"pred_vocal": [
                wandb.Audio(np.mean(separated_vocal["reverb_vocal"],axis=0), caption="reverb", sample_rate=self.h_params.preprocess.sample_rate),
                wandb.Audio(np.mean(separated_vocal["dry_vocal"],axis=0), caption="dry", sample_rate=self.h_params.preprocess.sample_rate)
            ]})
        
            #wandb.log({"wet_dry": torch.sigmoid(model.wet_dry)})