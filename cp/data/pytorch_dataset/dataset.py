import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass