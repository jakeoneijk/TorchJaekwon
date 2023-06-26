from typing import Any
from torch import Tensor
from numpy import ndarray

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class UtilTorch:
    @staticmethod
    def to_np(tensor:Tensor, do_squeeze:bool = True) -> ndarray:
        if do_squeeze:
            return tensor.squeeze().detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()
    
    @staticmethod
    def register_buffer_return_value(model:nn.Module,
                                     variable_name:str,
                                     value:Any):
        if type(value) != Tensor:
            value = torch.tensor(value, dtype=torch.float32)
        model.register_buffer(variable_name, value)
        return getattr(model,variable_name)
    
    @staticmethod
    def get_total_param_num(model:nn.Module) -> int:
        num_param : int = sum(param.numel() for param in model.parameters())
        return num_param
    
    @staticmethod
    def get_trainable_param_num(model:nn.Module) -> int:
        trainable_param : int = sum(param.numel() for param in model.parameters() if param.requires_grad)
        return trainable_param
    
    @staticmethod
    def tsne_plot(save_file_path:str,
                  class_array:ndarray, #[the number of data, 1] data must be integer for class. ex) [[1],[3],...]
                  embedding_array:ndarray,  #[the number of data, channel_size]
                  figure_size:tuple = (10,10)
                  ) -> None:
        assert os.path.splitext(save_file_path)[-1] == '.png', 'save_file_path should be *.png'

        print('generating t-SNE plot...')
        tsne = TSNE(random_state=0)
        tsne_output:ndarray = tsne.fit_transform(embedding_array)

        df = pd.DataFrame(tsne_output, columns=['x', 'y'])
        df['class'] = class_array

        plt.rcParams['figure.figsize'] = figure_size
        sns.scatterplot(
            x='x', y='y',
            hue='class',
            palette=sns.color_palette("hls", 10),
            data=df,
            marker='o',
            legend="full",
            alpha=0.5
        )

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        plt.savefig(save_file_path, bbox_inches='tight')