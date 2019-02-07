from pdb import set_trace

from collections import defaultdict, namedtuple
from pathlib import Path
import shutil

from typing import Tuple, List, Union

import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

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

    def on_train_begin(self, state):
        pass

    def on_train_end(self, state):
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

    def on_train_begin(self, state):
        for cb in self.callbacks:
            cb.on_train_begin()

    def on_train_end(self, state):
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

    def on_train_begin(self, state):
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
        if state.mode == "train":
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
            state.stop_epoch = True


class TBLogger(Callback):
    def __init__(self, logdir):
        self.logdir = Path(logdir)
        self.writer = None
        self.total_iter = 0
        self.train_iter = 0
        self.val_iter = 0
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def update_total_iter(self, mode):
        if mode == "train":
            self.train_iter += 1
        if mode == "test":
            self.val_iter += 1
        self.total_iter += 1

    def on_train_begin(self, state):
        self.train_iter = 0
        self.val_iter = 0
        self.logdir.mkdir(exist_ok=True)
        self.train_writer = SummaryWriter(str(self.logdir / "train"))
        self.val_writer = SummaryWriter(str(self.logdir / "val"))
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def on_batch_end(self, i, state):
        if state.mode == "train":
            self.update_total_iter("train")
            for name, metric in state.metrics["train"].items():
                self.train_writer.add_scalar(f"batch/{name}",
                                             float(metric),
                                             global_step=self.total_iter)
                self.train_metrics[name].append(float(metric))

            lr = get_opt_lr(state.opt)
            self.train_writer.add_scalar("lr",
                                         float(lr),
                                         global_step=self.train_iter)

        elif state.mode == "val":
            self.update_total_iter("val")
            for name, metric in state.metrics["val"].items():
                self.val_writer.add_scalar(f"batch/{name}",
                                           float(metric),
                                           global_step=self.total_iter)
                self.val_metrics[name].append(float(metric))

    def on_epoch_end(self, epoch, state):
        if state.mode == "train":
            for name, metric in self.train_metrics.items():
                mean = np.mean(metric[-10:])  # get last 10 values as approximation
                self.train_writer.add_scalar(f"epoch/{name}",
                                             float(mean),
                                             global_step=epoch)
        if state.mode == "val":
            for name, metric in self.train_metrics.items():
                mean = np.mean(metric)
                self.val_writer.add_scalar(f"epoch/{name}",
                                           float(mean),
                                           global_step=epoch)

    def on_train_end(self, state):
        self.train_writer.close()
        self.val_writer.close()


class SimpleLossCallback(Callback):
    def __init__(self, target_key, preds_key):
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


class SimpleSchedulerCallback(Callback):
    def __init__(self,
                 sched: Union[_LRScheduler, ReduceLROnPlateau],
                 metric: str = None):
        self.metric = metric or "val_loss"
        if isinstance(sched, ReduceLROnPlateau):
            self.when = "on_epoch_end"
        else:
            self.when = "on_epoch_begin"

    def on_epoch_begin(self, epoch, epochs, state):
        if self.when == "on_epoch_begin" and state.mode == "train":
            state.sched.step()

    def on_epoch_end(self, epoch, state):
        if self.when == "on_epoch_end" and state.mode == "train":
            state.sched.step(state.epoch_metrics[self.metric])


class ProgressBarCallback(Callback):
    def __init__(self):
        # self.pbar = None
        self.running_loss = None

    def on_epoch_begin(self, epoch, epochs, state):
        loader = state.loader
        if state.mode == "train":
            description = f"Epoch {epoch+1}/{epochs}"
            state.pbar = get_pbar(loader, description)
        elif state.mode == "test":
            description = "Predict"
            state.pbar = get_pbar(loader, description)

    def on_batch_end(self, i, state):

        if state.mode == "train":
            if not self.running_loss:
                self.running_loss = to_numpy(state.loss)

            self.running_loss = exp_weight_average(state.loss,
                                                   self.running_loss)
            postfix = {"loss": f"{self.running_loss:.4f}"}
            state.pbar.set_postfix(postfix)
            state.pbar.update()
        elif state.mode == "test":
            state.pbar.update()

    def on_epoch_end(self, epoch, state):
        if state.mode == "val":
            metrics = state.get("epoch_metrics", {})
            state.pbar.set_postfix_str(extend_postfix(state.pbar.postfix,
                                                      metrics))
            # set_trace()
            state.pbar.close()
        elif state.mode == "test":
            state.pbar.close()


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


class CheckpointSaverCallback(Callback):
    def __init__(self, savedir: str, metric: str = None, n_best: int = 3,
                 prefix: str = None, mode: str = "min"):
        self.metric = metric or "val_loss"
        self.n_best = n_best
        self.savedir = Path(savedir)
        self.prefix = prefix or "checkpoint."

        if mode not in ["min", "max"]:
            raise ValueError(f"mode should be 'min' or 'max', got {mode}")
        if mode == "min":
            self.maximize = False
        if mode == "max":
            self.maximize = True

        self.best_scores = []

    def on_epoch_begin(self, epoch, epochs, state):
        # trim best scores
        self.best_scores = self.best_scores[:epochs]

    def on_epoch_end(self, epoch, state):
        if state.mode == "val":
            score_val = state.epoch_metrics["val_loss"]
            score_name = f"{self.prefix}{epoch + 1}.h5"
            score = (score_val, score_name)
            sorted_scores = sorted(self.best_scores + [score],
                                   reverse=self.maximize)
            self.best_scores = sorted_scores[:self.n_best]
            if score_name in (s[1] for s in self.best_scores):
                state.checkpoint = f"{self.savedir / score_name}"
                # remove worst checkpoint
                if len(self.best_scores) > self.n_best:
                    Path(f"{self.savedir / sorted_scores[-1][1]}").unlink()

    def on_train_end(self, state):
        best_cp = self.savedir / self.best_scores[0][1]
        shutil.copy(str(best_cp), f"{self.savedir}/{self.prefix}best.h5")
        print(f"\nCheckpoint\t{self.metric or 'val_loss'}")
        for score in self.best_scores:
            print(f"{self.savedir/score[1]}\t{score[0]:.6f}")


class EarlyStoppingCallback(Callback):
    def __init__(self,
                 patience: int,
                 metric: str = None,
                 mode: str = "min",
                 min_delta: int = 0):
        self.best_score = None
        self.metric = metric or "val_loss"
        self.patience = patience
        self.num_bad_epochs = 0
        self.is_better = None

        if mode not in ["min", "max"]:
            raise ValueError(f"mode should be 'min' or 'max', got {mode}")
        if mode == "min":
            self.is_better = lambda score, best: score <= (best - min_delta)
        if mode == "max":
            self.is_better = lambda score, best: score >= (best - min_delta)

    def on_epoch_end(self, epoch, state):
        if state.mode == "val":
            score = state.epoch_metrics[self.metric]
            if self.best_score is None:
                self.best_score = score
            if self.is_better(score, self.best_score):
                self.num_bad_epochs = 0
                self.best_score = score
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                state.stop_train = True


class DebuggerCallback(Callback):
    def __init__(self, when: List[str], modes: List[str]):
        self.when = when
        self.modes = modes

    def on_batch_begin(self, i, state):
        if "on_batch_begin" in self.when:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_batch_end(self, i, state):
        if "on_batch_end" in self.when:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_begin(self, epoch, epochs, state):
        if "on_epoch_begin" in self.when:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_end(self, epoch, state):
        if "on_epoch_end" in self.when:
            if state.mode == "train" and "train" in self.modes:
                set_trace()
            if state.mode == "val" and "val" in self.modes:
                set_trace()
            if state.mode == "test" and "test" in self.modes:
                set_trace()

    def on_train_begin(self, state):
        if "on_train_begin" in self.when:
            set_trace()

    def on_train_end(self, state):
        if "on_train_end" in self.when:
            set_trace()
