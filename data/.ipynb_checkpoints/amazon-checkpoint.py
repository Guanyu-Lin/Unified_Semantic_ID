import torch
import json
from torch.utils.data import Dataset
# from torch_geometric.datasets import MovieLens1M
# import pandas as pd
import numpy as np
PROCESSED_MOVIE_LENS_SUFFIX = "/processed/item2text."


class AmazonData(Dataset):
    def __init__(
        self,
        data_name: str,
        *args,
        **kwargs
    ) -> None:

        item2attributes_text_file = 'data/'+ data_name + '_item2attributes_text.json'
        data_text_dict = json.loads(open(item2attributes_text_file).readline())

        self.ids = list(data_text_dict.keys())
        self.text =  list(data_text_dict.values())


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (torch.tensor(int(self.ids[idx]), dtype=torch.long), self.text[idx])
