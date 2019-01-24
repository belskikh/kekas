from pdb import set_trace as st

from collections import defaultdict
from pathlib import Path

import numpy as np

from tensorboardX import SummaryWriter

from .utils import get_opt_lr, get_pbar, \
    exp_weight_average, extend_postfix, to_numpy


class Callback:
    """
    Abstract base class used to build new callbacks.
    """
    def __init__(self):
        self.keker = None

    def init(self, keker):
        self.keker = keker

    def on_batch_begin(self, i, state):
        pass

    def on_batch_end(self, i, state):
        pass

    def on_epoch_begin(self, epoch, epochs, state):
        pass

    def on_epoch_end(self, epoch, state):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks:
    def __init__(self, callbacks, keker):
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

        for cb in self.callbacks:
            cb.init(keker)

    def on_batch_begin(self, i, state):
        for cb in self.callbacks:
            cb.on_batch_begin(i, state)

    def on_batch_end(self, i, state):
        for cb in self.callbacks:
            cb.on_batch_end(i, state)

    def on_epoch_begin(self, epoch, epochs, state):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, epochs, state)

    def on_epoch_end(self, epoch, state):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, state)

    def on_train_begin(self):
        for cb in self.callbacks:
            cb.on_train_begin()

    def on_train_end(self):
        for cb in self.callbacks:
            cb.on_train_end()


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""

    def __init__(self, init_lr):
        super().__init__()
        self.init_lr = init_lr

    def calc_lr(self):
        raise NotImplementedError

    def calc_momentum(self):
        raise NotImplementedError

    def update_lr(self, optimizer):
        new_lr = self.calc_lr()
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def update_momentum(self, optimizer):
        new_momentum = self.calc_momentum()
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum
        return new_momentum

    def on_batch_begin(self, i, state):
        if state.mode == "train":
            self.update_lr(state.opt)
            self.update_momentum(state.opt)


class OneCycleLR(LRUpdater):
    """
    An learning rate updater
        that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    https://github.com/Scitator/pytorch-common/blob/master/train/callbacks.py
    """

    def __init__(self, init_lr, cycle_len, div, cut_div, momentum_range, len_loader):
        """
        :param init_lr: init learning rate for torch optimizer
        :param cycle_len: (int) num epochs to apply one cycle policy
        :param div: (int) ratio between initial lr and maximum lr
        :param cut_div: (int) which part of cycle lr will grow
            (Ex: cut_div=4 -> 1/4 lr grow, 3/4 lr decrease
        :param momentum_range: (tuple(int, int)) max and min momentum values
        """
        super().__init__(init_lr)
        self.len_loader = len_loader
        self.total_iter = None
        self.div = div
        self.cut_div = cut_div
        self.cycle_iter = 0
        self.cycle_count = 0
        self.cycle_len = cycle_len
        # point in iterations for starting lr decreasing
        self.cut_point = None
        self.momentum_range = momentum_range

    def calc_lr(self):
        # calculate percent for learning rate change
        if self.cycle_iter > self.cut_point:
            percent = 1 - (self.cycle_iter - self.cut_point) / (
                    self.total_iter - self.cut_point)
        else:
            percent = self.cycle_iter / self.cut_point
        res = self.init_lr * (1 + percent * (self.div - 1)) / self.div

        self.cycle_iter += 1
        if self.cycle_iter == self.total_iter:
            self.cycle_iter = 0
            self.cycle_count += 1
        return res

    def calc_momentum(self):
        if self.cycle_iter > self.cut_point:
            percent = (self.cycle_iter - self.cut_point) / (self.total_iter - self.cut_point)
        else:
            percent = 1 - self.cycle_iter / self.cut_point
        res = self.momentum_range[1] + percent * (
                self.momentum_range[0] - self.momentum_range[1])
        return res

    def on_train_begin(self):
        self.total_iter = self.len_loader * self.cycle_len
        self.cut_point = self.total_iter // self.cut_div


class LRFinder(LRUpdater):
    """
    Helps you find an optimal learning rate for a model,
        as per suggetion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input.

    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self, final_lr, n_steps=None, optimizer_key="main"):
        """

        :param init_lr: initial learning rate to use
        :param final_lr: final learning rate to try with
        :param n_steps:  number of batches to try;
            if None - whole loader would be used.
        :param optimizer_key: which optimizer key to use
            for learning rate scheduling
        """
        super().__init__()

        self.final_lr = final_lr
        self.n_steps = n_steps
        self.multiplier = 0
        self.find_iter = 0

    def calc_lr(self):
        res = self.init_lr * self.multiplier ** self.find_iter
        self.find_iter += 1
        return res

    def on_batch_end(self, i, state):
        super().on_batch_end(i, state)
        if self.find_iter > self.n_steps:
            raise NotImplementedError("End of LRFinder")


class TBLogger(Callback):
    # TODO: outdated, fix it
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self):
        Path(self.log_dir).mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_batch_end(self, i, state):
        if state.mode == "train":
            lr = get_opt_lr(state.opt)
            loss = state.loss.value

            self.writer.add_scalar(f"")

    def on_epoch_end(self, epoch, epoch_metrics):

        for k, v in self.metrics.train_metrics.items():
            self.writer.add_scalar(f'train/{k}', float(v), global_step=epoch)

        for k, v in self.metrics.val_metrics.items():
            self.writer.add_scalar(f'val/{k}', float(v), global_step=epoch)

        for idx, param_group in enumerate(self.runner.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'group{idx}/lr', float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()


class SimpleLossCallback(Callback):
    def __init__(self, target_key, preds_key):
        super().__init__()
        self.target_key = target_key
        self.preds_key = preds_key

    def on_batch_end(self, i, state):
        target = state.batch[self.target_key]
        preds = state.out[self.preds_key]

        state.loss = state.criterion(preds, target)


class SimpleOptimizerCallback(Callback):
    def on_batch_end(self, i, state):
        if state.mode == "train":
            state.opt.zero_grad()
            state.loss.backward()
            state.opt.step()


class ProgressBarCallback(Callback):
    # TODO: change mode settings 'test' -> 'predict'
    # to make possible predicting on loader
    def __init__(self):
        super().__init__()
        self.pbar = None
        self.running_loss = None

    def on_epoch_begin(self, epoch, epochs, state):
        loader = state.loader
        if state.mode == "train":
            description = f"Epoch {epoch+1}/{epochs}"
            self.pbar = get_pbar(loader, description)
        elif state.mode == "test":
            description = "Predict"
            self.pbar = get_pbar(loader, description)
        self.running_loss = 0.0

    def on_batch_end(self, i, state):
        if state.mode == "train":
            self.running_loss = exp_weight_average(state.loss,
                                                   self.running_loss)
            postfix = {"loss": f"{self.running_loss:.4f}"}
            self.pbar.set_postfix(postfix)
            self.pbar.update()
        elif state.mode == "test":
            self.pbar.update()

    def on_epoch_end(self, epoch, state):
        if state.mode == "val":
            metrics = state.get("epoch_metrics", {})
            self.pbar.set_postfix_str(extend_postfix(self.pbar.postfix,
                                                     metrics))
            self.pbar.close()
        elif state.mode == "test":
            self.pbar.close()


class MetricsCallback(Callback):
    # TODO: split loss and metrics
    def __init__(self, metrics, target_key, preds_key):
        super().__init__()
        self.metrics = metrics
        self.epoch_metrics = None
        self.target_key = target_key
        self.preds_key = preds_key

    def update_epoch_metrics(self, target, preds):
        for m in self.metrics:
            value = m(target, preds)
            self.epoch_metrics[m.__name__] += value

    def on_epoch_begin(self, epoch, epochs, state):
        self.epoch_metrics = defaultdict(float)

    def on_batch_end(self, i, state):
        if state.mode == "val":
            self.epoch_metrics["val_loss"] += to_numpy(state.loss)
            self.update_epoch_metrics(target=state.batch[self.target_key],
                                      preds=state.out[self.preds_key])

    def on_epoch_end(self, epoch, state):
        divider = len(state.loader)
        for k in self.epoch_metrics.keys():
            self.epoch_metrics[k] /= divider

        state.epoch_metrics = self.epoch_metrics


class PredictionsSaverCallback(Callback):
    def __init__(self, savepath, preds_key):
        super().__init__()
        self.savepath = savepath
        self.preds_key = preds_key
        self.preds = []

    def on_batch_end(self, i, state):
        if state.mode == "test":
            self.preds.append(to_numpy(state.out[self.preds_key]))

    def on_epoch_end(self, epoch, state):
        if state.mode == "test":
            np.save(self.savepath, np.concatenate(self.preds))
