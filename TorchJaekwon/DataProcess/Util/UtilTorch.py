from typing import Any
from torch import Tensor
from numpy import ndarray

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
                  figure_size:tuple = (10,10),
                  legend:str = 'full',
                  point_size:float = None #s=200
                  ) -> None:
        import pandas as pd
        import seaborn as sns
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
            legend=legend,
            alpha=0.5,
            s = point_size
        )

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        plt.savefig(save_file_path, bbox_inches='tight')
    
    @staticmethod
    def interpolate_2d(input:Tensor, #[width, height] | [batch, width, height] | [batch, channels, width, height]
                       size_after_interpolation:tuple, #(width, height)
                       mode:str = 'nearest'
                       ) -> Tensor:
        if len(input.shape) == 2:
            shape_after_interpolation = size_after_interpolation
            input = input.view(1,1,*(input.shape))
        elif len(input.shape) == 3:
            shape_after_interpolation = (input.shape[0],*(size_after_interpolation))
            input = input.unsqueeze(1)
        elif len(input.shape) == 4:
            shape_after_interpolation = (input.shape[0],input.shape[1],*(size_after_interpolation))
        return F.interpolate(input, size = size_after_interpolation, mode=mode).view(shape_after_interpolation)