from pdb import set_trace as st

from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD

from .callbacks import Callbacks, ProgressBarCallback, SimpleOptimizerCallback
from .data import DataOwner
from .utils import exp_weight_average, get_pbar, \
    to_numpy, update_epoch_metrics, extend_postfix, DotDict
from .state import State


class Keker:
    ''' The core class that proivdes main methods for training and
    predicting on given model and dataset
    '''
    def __init__(self, model, dataowner,
                 opt_fn=None, criterion=None,
                 device=None,
                 callbacks=None):
        assert isinstance(dataowner, DataOwner), "I need DataOwner, human"

        self.model = model
        self.dataowner = dataowner
        self.opt_fn = opt_fn or partial(SGD)
        self.device = device or torch.device("cuda" if
                                             torch.cuda.is_available()
                                             else "cpu")
        callbacks = callbacks + [SimpleOptimizerCallback(),
                                 ProgressBarCallback()]
        self.callbacks = Callbacks(callbacks, self)
        self.state = DotDict()
        self.state.criterion = criterion

    def kek(self, lr, epochs):
        self.state.opt = self.opt_fn(params=self.model.parameters(), lr=lr)

        self.model.to(self.device)

        self.callbacks.on_train_begin()

        for epoch in range(epochs):
            self.set_mode("train")
            self._run_epoch(epoch, epochs)

            self.set_mode("val")
            self._run_epoch(epoch, epochs)

        self.callbacks.on_train_end()

    def _run_epoch(self, epoch, epochs):
        self.callbacks.on_epoch_begin(epoch, epochs, self.state)

        with torch.set_grad_enabled(self.is_train):
            for i, batch in enumerate(self.state.loader):
                self.callbacks.on_batch_begin(i, self.state)

                self.state.batch = self.to_device(batch)

                self.state.out = self.step()

                self.callbacks.on_batch_end(i, self.state)

        self.callbacks.on_epoch_end(epoch, self.state)

    def step(self):
        inp = self.state.batch["image"]
        logits = self.model(inp)

        return {"logits": logits}

    def predict_test(self):
        self.set_mode("test")
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

    def predict_loader(self, loader):
        self.state.loader = loader
        self.model.eval()
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

    def TTA(self, n_aug=6):
        pass

    def save(self, savepath):
        savepath = Path(savepath)
        savepath.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), savepath)

    def load(self, loadpath):
        self.model.load_state_dict(torch.load(loadpath))

    def to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()
                if hasattr(v, "to")}

    def set_mode(self, mode):
        if mode == "train":
            self.model.train()
            self.state.loader = self.dataowner.train_dl
        elif mode == "val":
            self.model.eval()
            self.state.loader = self.dataowner.val_dl
        elif mode == "test":
            self.model.eval()
            self.state.loader = self.dataowner.test_dl
        self.state.mode = mode

    @property
    def is_train(self):
        return self.state.mode == "train"
