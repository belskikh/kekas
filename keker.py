from pdb import set_trace as st

from functools import partial

from tqdm import tqdm, tnrange

import torch
from torch.optim import SGD

from .data import DataOwner

class Keker:
    ''' The core class that proivdes main methods for training and
    predicting on given model and dataset
    '''
    def __init__(self, model, dataowner,
                 opt_fn=None, criterion=None, metrics=None, device=None):
        assert isinstance(dataowner, DataOwner), "I need DataOwner, human"

        self.model = model
        self.dataowner = dataowner
        self.opt_fn = opt_fn or partial(SGD)
        self.criterion = criterion
        self.metrics = metrics or []
        self.device = device or torch.device("cuda" if
                                             torch.cuda.is_available()
                                             else "cpu")

    def kek(self, lr, epochs):
        batch_num = 0

        opt = self.opt_fn(params=self.model.parameters(), lr=lr)

        self.model.to(self.device)  # TODO: move to init?

        for epoch in tnrange(epochs, desc='Epoch'):
            train_dl_iter = tqdm(iter(self.dataowner.train_dl), leave=False,
                           total=len(self.dataowner.train_dl), miniters=0)

            # TODO: batch hanlder
            with torch.set_grad_enabled(True):
                self.model.train()
                for elem in train_dl_iter:
                    batch_num += 1
                    # st()
                    inp = {k: v.to(self.device) for k, v in elem.items()}
                    try:
                        out = self.model(inp["image"])
                    except:
                        st()

                    # TODO: loss to callback
                    loss = self.criterion(out, inp["label"])
                    loss.backward()

                    # TODO: optimizer to callback
                    opt.step()
                    opt.zero_grad()

                    # TODO: metrics to callback

            val_dl_iter = iter(self.dataowner.val_dl)
            for elem in val_dl_iter:
                self.model.eval()
                inp = {k: v.to(self.device) for k, v in elem.items()}
                out = self.model(inp["image"])
                loss = self.criterion(out, inp["label"])

                metrics = []






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


class Stepper():
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # self.reset(True)

    # def reset(self, train=True):
    #     if train: apply_leaf(self.m, set_train_mode)
    #     else: self.m.eval()
    #     if hasattr(self.m, 'reset'):
    #         self.model.reset()


    def step(self, inp, target):

        output = self.model(*inp)


    # def evaluate(self, xs, y):
    #     preds = self.m(*xs)
    #     if isinstance(preds,tuple): preds=preds[0]
    #     return preds, self.crit(preds, y)