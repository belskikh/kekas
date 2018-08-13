import torch
from torch.optim import Adam

class Keker():
    ''' The core class that proivdes main methods for training and predicting on given model and dataset
    '''
    def __init__(self, model, train_dl, val_dl=None, test_dl=None,
                 opt_fn=None, criterion=None, metrics=None):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.opt_fn = opt_fn or Adam(1e-3)
        self.criterion = criterion
        self.metrics = metrics or []


    def kek(self, lr, epochs, one_cycle=(10, 2, 0.95, 0.85), w_decay=10e-5,
            stepper=None, callbacks=None):
        pass

    def lr_range_test(self, start_lr=1e-5, stop_lr=1e-2, n_iter=None, w_decay=None, linear=False,):
        pass

    def predict(self, dl=None):
        dl = dl or self.test_dl
        pass

    def TTA(self, n_aug=6):
        pass

    def save(self, path):
        pass

    def load(self, weights):
        pass

