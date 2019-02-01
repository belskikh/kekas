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
    PredictionsSaverCallback, OneCycleLR, SimpleLossCallback, MetricsCallback, \
    TBLogger, LRFinder
from .data import DataOwner
from .parallel import DataParallelCriterion, DataParallelModel
from .utils import DotDict


class Keker:
    ''' The core class that proivdes main methods for training and
    predicting on given model and dataset
    '''
    def __init__(self, model, dataowner,
                 target_key="label", preds_key="logits",
                 metrics=None,
                 opt_fn=None, criterion=None,
                 device=None, step_fn=None,
                 loss_cb=None, opt_cb=None, callbacks=None,
                 early_stop=None, tb_logdir=None):
        assert isinstance(dataowner, DataOwner), "I need DataOwner, human"

        self.state = DotDict()

        self.model = model

        self.target_key = target_key
        self.preds_key = preds_key

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

        loss_cb = loss_cb or SimpleLossCallback(target_key, preds_key)
        opt_cb = opt_cb or SimpleOptimizerCallback()
        metrics_cb = MetricsCallback(target_key, preds_key, metrics)

        self.core_callbacks = callbacks or [] + [loss_cb,
                                                 metrics_cb,
                                                 opt_cb,
                                                 ProgressBarCallback()]
        callbacks = self.core_callbacks[:]

        if tb_logdir:
            self.state.do_log = True
            self.state.metrics = defaultdict(dict)
            callbacks += [TBLogger(tb_logdir)]
            self.tb_logdir = tb_logdir

        self.callbacks = Callbacks(callbacks)

        self.early_stop = early_stop

    def kek(self, lr, epochs, skip_val=False):
        self.state.opt = self.opt_fn(params=self.model.parameters(), lr=lr)

        self.callbacks.on_train_begin()

        for epoch in range(epochs):
            self.set_mode("train")
            self._run_epoch(epoch, epochs)

            if not skip_val:
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
        one_cycle_cb = OneCycleLR(max_lr, cycle_len, len_loader,
                                  momentum_range, div_factor, increase_fraction)

        self.callbacks = Callbacks(callbacks.callbacks + [one_cycle_cb])

        self.kek(lr=max_lr, epochs=cycle_len)

        # set old callbacks without OneCycle
        self.callbacks = callbacks

    def kek_lr(self,
               final_lr,
               init_lr: float=1e-6,
               n_steps: int=None,
               logdir: str=None):

        logdir = logdir or Path(self.tb_logdir) / "lr_find"
        tmp_path = Path(logdir) / "tmp"
        self.save(str(tmp_path) + "/tmp.h5")

        callbacks = self.callbacks

        try:
            # temporarily add Logger and LRFinder callback
            tblogger_cb = TBLogger(str(logdir))
            lrfinder_cb = LRFinder(final_lr, init_lr, n_steps)

            self.callbacks = Callbacks(self.core_callbacks + [tblogger_cb,
                                                              lrfinder_cb])

            self.kek(lr=init_lr, epochs=1, skip_val=True)
        finally:
            # set old callbacks without LRFinder
            self.callbacks = callbacks
            self.load(str(tmp_path) + "/tmp.h5")

    def _run_epoch(self, epoch, epochs):
        self.callbacks.on_epoch_begin(epoch, epochs, self.state)

        with torch.set_grad_enabled(self.is_train):
            for i, batch in enumerate(self.state.loader):
                if self.early_stop:
                    if self.state.mode == "train":
                        if i > self.early_stop:
                            break
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

    def predict_loader(self, loader, savepath):
        callbacks = self.callbacks

        tmp_callbacks = Callbacks([ProgressBarCallback(),
                                   PredictionsSaverCallback(savepath,
                                                            self.preds_key)])

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
