import torch
from utils.classes import Singleton


class MovieDataloader(torch.utils.data.Dataset, metaclass=Singleton):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
