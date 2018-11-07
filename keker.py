from pdb import set_trace as st

from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD

from .callbacks import Callbacks
from .data import DataOwner
from .utils import exp_weight_average, get_trainval_pbar, get_predict_pbar, \
    to_numpy, update_epoch_metrics, extended_postfix
from .state import State


class Keker:
    ''' The core class that proivdes main methods for training and
    predicting on given model and dataset
    '''
    def __init__(self, model, dataowner,
                 opt_fn=None, criterion=None,
                 metrics=None, device=None,
                 callbacks=None):
        assert isinstance(dataowner, DataOwner), "I need DataOwner, human"

        self.model = model
        self.dataowner = dataowner
        self.opt_fn = opt_fn or partial(SGD)
        self.criterion = criterion
        self.metrics = metrics or []
        self.device = device or torch.device("cuda" if
                                             torch.cuda.is_available()
                                             else "cpu")
        self.callbacks = Callbacks(callbacks) or None
        self.mode = None
        self.state = State()

    def kek(self, lr, epochs):
        self.opt = self.opt_fn(params=self.model.parameters(), lr=lr)

        self.model.to(self.device)

        self.callbacks.on_train_begin()

        for epoch in range(epochs):

            pbar = get_trainval_pbar(
                dataloader=self.dataowner.train_dl,
                epoch=epoch,
                epochs=epochs)

            self.set_mode("train")
            self._run_epoch(epoch, self.dataowner.train_dl, pbar)

            self.set_mode("val")
            self._run_epoch(epoch, self.dataowner.val_dl, pbar)

            pbar.close()

        self.callbacks.on_train_end()

    def _run_epoch(self, epoch, loader, pbar):
        self.callbacks.on_epoch_begin(epoch)
        is_train = self.state.is_train
        running_loss = 0.0
        epoch_metrics = defaultdict(float)
        with torch.set_grad_enabled(is_train):
            for i, batch in enumerate(loader):
                self.callbacks.on_batch_begin(i, self.state)

                batch = self.to_device(batch)

                self.step(batch)

                if is_train:
                    loss = self.state.loss
                    # epoch_metrics["trn_loss"] += to_numpy(self.state.loss)
                    # update postfix
                    running_loss = exp_weight_average(loss, running_loss)
                    postfix = {"loss": f"{running_loss:.4f}"}
                    pbar.set_postfix(postfix)
                    pbar.update()
                else:
                    epoch_metrics["val_loss"] += to_numpy(self.state.loss)
                    update_epoch_metrics(
                        target=self.state.target,
                        preds=self.state.out,
                        metrics=self.metrics,
                        epoch_metrics=epoch_metrics
                    )

                self.callbacks.on_batch_end(i, self.state)

            # average metrics
            for k, v in epoch_metrics.items():
                epoch_metrics[k] /= len(loader)

            # update pbar
            pbar.set_postfix_str(extended_postfix(pbar.postfix, epoch_metrics))

        self.callbacks.on_epoch_end(epoch, epoch_metrics)

    def step(self, batch):
        report = {}
        inp = batch["image"]
        out = self.model(inp)

        if self.mode != "test":
            target = batch["label"]
            loss = self.criterion(self.state.out, target)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            report["target"] = target
            report["loss"] = loss.data

        report["inp"] = inp
        report["out"] = out


        return report

    def calc_loss(self, batch):
        target = batch["label"]
        loss = self.criterion(self.state.out, target)

        self.state.update(target=target, loss=loss)

    def predict(self, dl=None):
        dl = dl or self.dataowner.test_dl
        pbar = get_predict_pbar(dl)
        preds = []
        with torch.set_grad_enabled(False):
            self.set_mode("test")
            for i, batch in enumerate(dl):
                batch = self.to_device(batch)
                self.step(batch)
                preds.append(to_numpy(self.state.out))
                pbar.update()
        return np.concatenate(preds)

    def TTA(self, n_aug=6):
        pass

    def save(self, savepath):
        savepath = Path(savepath)
        savepath.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), savepath)

    def load(self, loadpath):
        self.model.load_state_dict(torch.load(loadpath))

    def to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def set_mode(self, mode):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        self.mode = mode

    def update_pbar(self):
        pass


