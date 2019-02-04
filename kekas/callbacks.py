from pdb import set_trace

from collections import defaultdict
from pathlib import Path
import shutil

from typing import Tuple, List

import numpy as np

from tensorboardX import SummaryWriter

from .utils import get_opt_lr, get_pbar, \
    exp_weight_average, extend_postfix, to_numpy


class Callback:
    """
    Abstract base class used to build new callbacks.
    """
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
    def __init__(self, callbacks):
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

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
    Inspired by
    https://github.com/fastai/fastai/blob/master/fastai/callbacks/one_cycle.py
    """
    def __init__(self,
                 max_lr: float,
                 cycle_len: int,
                 len_loader: int,
                 momentum_range: Tuple[float, float],
                 div_factor: float,
                 increase_fraction: float) -> None:
        super().__init__(max_lr)
        self.cycle_len = cycle_len
        self.momentum_range = momentum_range
        self.div_factor = div_factor
        self.increase_fraction = increase_fraction
        self.len_loader = len_loader
        self.total_iter = None
        self.cycle_iter = 0
        # point in iterations for starting lr decreasing
        self.cut_point = None

    def on_train_begin(self):
        self.total_iter = self.len_loader * self.cycle_len
        self.cut_point = int(self.total_iter * self.increase_fraction)

    def calc_lr(self):
        # calculate percent for learning rate change
        if self.cycle_iter <= self.cut_point:
            percent = self.cycle_iter / self.cut_point

        else:
            percent = 1 - (self.cycle_iter - self.cut_point) / (
                    self.total_iter - self.cut_point)

        res = self.init_lr * (1 + percent * (self.div_factor - 1)) / self.div_factor

        return res

    def calc_momentum(self):
        if self.cycle_iter <= self.cut_point:
            percent = 1 - self.cycle_iter / self.cut_point

        else:
            percent = (self.cycle_iter - self.cut_point) / (self.total_iter - self.cut_point)
        res = self.momentum_range[1] + percent * (
                self.momentum_range[0] - self.momentum_range[1])
        return res

    def on_batch_begin(self, i, state):
        super().on_batch_begin(i, state)
        self.cycle_iter += 1


class LRFinder(LRUpdater):
    """
    Helps you find an optimal learning rate for a model,
        as per suggetion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input.

    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self, final_lr, init_lr=1e-6, n_steps=None):
        super().__init__(init_lr)
        self.final_lr = final_lr
        self.n_steps = n_steps
        self.multiplier = 0
        self.n = 0

    def calc_lr(self):
        res = self.init_lr * (self.final_lr / self.init_lr) ** \
              (self.n / self.n_steps)
        self.n += 1
        return res

    def calc_momentum(self):
        pass

    def update_momentum(self, optimizer):
        pass

    def on_epoch_begin(self, epoch, epochs, state):
        self.multiplier = self.init_lr ** (1 / self.n_steps)

    def on_batch_end(self, i, state):
        super().on_batch_end(i, state)
        if self.n > self.n_steps:
            print("End of LRFinder")
            state.stop = True


class TBLogger(Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None
        self.train_iter = 0
        self.val_iter = 0

    def update_total_iter(self, mode):
        if mode == "train":
            self.train_iter += 1
        if mode == "val":
            self.val_iter += 1

    def on_train_begin(self):
        Path(self.log_dir).mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_batch_end(self, i, state):
        if state.mode == "train":
            self.update_total_iter("train")
            for name, metric in state.metrics["train"].items():
                self.writer.add_scalar(f"train/{name}",
                                       float(metric),
                                       global_step=self.train_iter)

            lr = get_opt_lr(state.opt)
            self.writer.add_scalar("train/lr",
                                   float(lr),
                                   global_step=self.train_iter)

        elif state.mode == "val":
            self.update_total_iter("val")
            for name, metric in state.metrics["val"].items():
                self.writer.add_scalar(f"val/{name}",
                                       float(metric),
                                       global_step=self.val_iter)

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

    def on_batch_end(self, i, state):

        if state.mode == "train":
            if not self.running_loss:
                self.running_loss = to_numpy(state.loss)

            self.running_loss = exp_weight_average(state.loss,
                                                   self.running_loss)
            postfix = {"loss": f"{self.running_loss:.4f}"}
            self.pbar.set_postfix(postfix)
            self.pbar.update()
        elif state.mode == "test":
            self.pbar.update()

    def on_epoch_end(self, epoch, state):
        if state.mode == "val":
            metrics = state.get("pbar_metrics", {})
            self.pbar.set_postfix_str(extend_postfix(self.pbar.postfix,
                                                     metrics))
            self.pbar.close()
        elif state.mode == "test":
            self.pbar.close()


class MetricsCallback(Callback):
    def __init__(self, target_key, preds_key, metrics=None):
        self.metrics = metrics or {}
        self.pbar_metrics = None
        self.target_key = target_key
        self.preds_key = preds_key

    def update_epoch_metrics(self, target, preds):
        for name, m in self.metrics.items():
            value = m(target, preds)
            self.pbar_metrics[name] += value

    def on_epoch_begin(self, epoch, epochs, state):
        self.pbar_metrics = defaultdict(float)

    def on_batch_end(self, i, state):
        if state.mode == "val":
            self.pbar_metrics["val_loss"] += float(to_numpy(state.loss))
            self.update_epoch_metrics(target=state.batch[self.target_key],
                                      preds=state.out[self.preds_key])
        # tb logs
        if state.mode != "test" and state.do_log:
            state.metrics[state.mode]["loss"] = float(to_numpy(state.loss))
            for name, m in self.metrics.items():
                value = m(target=state.batch[self.target_key],
                          preds=state.out[self.preds_key])
                state.metrics[state.mode][name] = value

    def on_epoch_end(self, epoch, state):
        divider = len(state.loader)
        for k in self.pbar_metrics.keys():
            self.pbar_metrics[k] /= divider
        state.epoch_metrics = self.pbar_metrics


class PredictionsSaverCallback(Callback):
    def __init__(self, savepath, preds_key):
        super().__init__()
        self.savepath = Path(savepath)
        self.savepath.parent.mkdir(exist_ok=True)
        self.preds_key = preds_key
        self.preds = []

    def on_batch_end(self, i, state):
        if state.mode == "test":
            out = state.out[self.preds_key]
            # DataParallelModel workaround
            if isinstance(out, list):
                out = np.concatenate([to_numpy(o) for o in out])
            self.preds.append(out)

    def on_epoch_end(self, epoch, state):
        if state.mode == "test":
            np.save(self.savepath, np.concatenate(self.preds))
            self.preds = []


class DebuggerCallback(Callback):
    def __init__(self, where: List[str], modes: List[str]):
        self.where = where
        self.modes = modes

    def on_batch_begin(self, i, state):
        if "on_batch_begin" in self.where:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_batch_end(self, i, state):
        if "on_batch_end" in self.where:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_begin(self, epoch, epochs, state):
        if "on_epoch_begin" in self.where:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_end(self, epoch, state):
        if "on_epoch_end" in self.where:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_train_begin(self):
        if "on_train_begin" in self.where:
            set_trace()

    def on_train_end(self):
        if "on_train_end" in self.where:
            set_trace()
