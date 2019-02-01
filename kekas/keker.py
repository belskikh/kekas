from pdb import set_trace as st

from collections import defaultdict
from functools import partial
from pathlib import Path

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from .callbacks import Callbacks, ProgressBarCallback, SimpleOptimizerCallback, \
    PredictionsSaverCallback, OneCycleLR
from .data import DataOwner
from .parallel import DataParallelCriterion, DataParallelModel
from .utils import DotDict


class Keker:
    ''' The core class that proivdes main methods for training and
    predicting on given model and dataset
    '''
    def __init__(self, model, dataowner,
                 opt_fn=None, criterion=None,
                 device=None, step_fn=None,
                 callbacks=[]):
        assert isinstance(dataowner, DataOwner), "I need DataOwner, human"

        self.state = DotDict()

        self.model = model
        self.state.criterion = criterion
        if torch.cuda.device_count() > 1:
            self.model = DataParallelModel(self.model)
            self.state.criterion = DataParallelCriterion(self.state.criterion)

        self.dataowner = dataowner
        self.opt_fn = opt_fn or partial(SGD)
        self.device = device or torch.device("cuda" if
                                             torch.cuda.is_available()
                                             else "cpu")
        self.model.to(self.device)

        self.step = step_fn or self.default_step

        callbacks = callbacks + [SimpleOptimizerCallback(),
                                 ProgressBarCallback()]
        self.callbacks = Callbacks(callbacks)

    def kek(self, lr, epochs):
        self.state.opt = self.opt_fn(params=self.model.parameters(), lr=lr)

        self.callbacks.on_train_begin()

        for epoch in range(epochs):
            self.set_mode("train")
            self._run_epoch(epoch, epochs)

            self.set_mode("val")
            self._run_epoch(epoch, epochs)

        self.callbacks.on_train_end()

    def kek_one_cycle(self,
                      max_lr: float,
                      cycle_len: int,
                      momentum_range: Tuple[float, float]=(0.95, 0.85),
                      div_factor: float=25,
                      increase_fraction: float=0.3) -> None:

        callbacks = self.callbacks

        # temporarily add OneCycle callback
        len_loader = len(self.dataowner.train_dl)
        one_cycle_clb = OneCycleLR(max_lr, cycle_len, len_loader,
                                   momentum_range, div_factor, increase_fraction)

        self.callbacks = Callbacks(callbacks.callbacks + [one_cycle_clb])

        self.kek(lr=max_lr, epochs=cycle_len)

        # set old callbacks without OneCycle
        self.callbacks = callbacks

    def _run_epoch(self, epoch, epochs):
        self.callbacks.on_epoch_begin(epoch, epochs, self.state)

        with torch.set_grad_enabled(self.is_train):
            for i, batch in enumerate(self.state.loader):
                self.callbacks.on_batch_begin(i, self.state)

                self.state.batch = self.to_device(batch)

                self.state.out = self.step()

                self.callbacks.on_batch_end(i, self.state)

        self.callbacks.on_epoch_end(epoch, self.state)

    def default_step(self):
        inp = self.state.batch["image"]
        logits = self.model(inp)

        return {"logits": logits}

    def predict(self):
        self.set_mode("test")
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

    def predict_loader(self, loader, savepath, preds_key):
        callbacks = self.callbacks

        tmp_callbacks = Callbacks([ProgressBarCallback(),
                                   PredictionsSaverCallback(savepath, preds_key)])

        self.callbacks = tmp_callbacks

        self.state.mode = "test"
        self.state.loader = loader
        self.model.eval()
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

        self.callbacks = callbacks

    def predict_array(self, array):
        array = torch.from_numpy(array).to(self.device)
        with torch.set_grad_enabled(False):
            preds = self.model(array)
        return preds

    def TTA(self, n_aug=6):
        # TODO: move to utils
        pass

    def save(self, savepath):
        savepath = Path(savepath)
        savepath.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), savepath)
        print(f"Weights saved to {savepath}")

    def load(self, loadpath):
        # TODO: find more elegant fix
        checkpoint = torch.load(loadpath,
                                map_location=lambda storage, loc: storage)
        if not isinstance(self.model, DataParallelModel) \
                and "module." in list(checkpoint.keys())[0]:
            # [7:] is to skip 'module.' in group name
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        print(f"Weights loaded from {loadpath}")

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
