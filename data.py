from collections import namedtuple

from torch.utils.data import DataLoader, Dataset


class DataKek(Dataset):
    """The easiest way to store the dataset info is pandas DataFrame.
    So DataKek gets a DataFrame as a source of data"""
    def __init__(self, df, reader_fn=None, transforms=None):
        # transform df to list
        self.data = list(df.iterrows())  # for multiprocessing
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __getitem__(self, ind):
        datum = self.reader_fn(*self.data[ind])
        if self.transforms is not None:
            datum = self.transforms(datum)
        return datum

    def __len__(self):
        return len(self.data)


DataOwner = namedtuple("DataOwner", ["train_dl", "val_dl", "test_dl"])
