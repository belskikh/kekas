from collections import namedtuple
from typing import Dict, Callable, Optional
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import Compose


class DataKek(Dataset):
    """The easiest way to store the dataset info is pandas DataFrame.
    So DataKek gets a DataFrame as a source of data"""
    def __init__(self,
                 df: pd.DataFrame,
                 reader_fn: Callable,
                 transforms: Optional[Compose] = None) -> None:
        # transform df to list
        self.data = list(df.values)  # for multiprocessing
        self.df_columns = df.columns
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __getitem__(self, ind: int) -> Dict:
        data_dict = {col: v for col, v in zip(self.df_columns, self.data[ind])}
        datum = self.reader_fn(ind, data_dict)
        if self.transforms is not None:
            datum = self.transforms(datum)
        return datum

    def __len__(self) -> int:
        return len(self.data)


DataOwner = namedtuple("DataOwner", ["train_dl", "val_dl", "test_dl"])
