from torch.utils.data import DataLoader, Dataset


class DataKek(Dataset):
    """The easiest way to store the dataset info is pandas dataframe. So DataKek gets a dataframe as src of data"""
    def __init__(self, df, transforms=None):
        # transform df to list
        self.data = [row for i, row in df.iterrows()]
        self.transforms = transforms
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.data)


class DataKeker(DataLoader):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __iter__(self):
        pass

    def __len__(self):
        pass
