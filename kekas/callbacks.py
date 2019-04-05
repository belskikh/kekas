from pdb import set_trace

from collections import defaultdict, namedtuple
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, Tuple, Type, List, Optional, Union
import warnings

try:
    from apex import amp
except:
    pass  # warning message appears in keker.py module, no needs to be here

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tensorboardX import SummaryWriter
from .utils import get_opt_lr, get_pbar, DotDict, \
    exp_weight_average, extend_postfix, to_numpy


class Callback:
    """
    Abstract base class used to build new callbacks.
    """
    def on_batch_begin(self, i: int, state: DotDict) -> None:
        pass

    def on_batch_end(self, i: int, state: DotDict) -> None:
        pass

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        pass

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        pass

    def on_train_begin(self, state: DotDict) -> None:
        pass

    def on_train_end(self, state: DotDict) -> None:
        pass


class Callbacks:
    def __init__(self, callbacks: Union[List, Any]) -> None:
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def on_batch_begin(self, i: int, state: DotDict) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(i, state)

    def on_batch_end(self, i: int, state: DotDict) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(i, state)

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, epochs, state)

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, state)

    def on_train_begin(self, state: DotDict) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: DotDict) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""
    def __init__(self, init_lr: float) -> None:
        self.init_lr = init_lr

    def calc_lr(self) -> float:
        raise NotImplementedError

    def calc_momentum(self) -> float:
        raise NotImplementedError

    def update_lr(self,
                  optimizer: torch.optim.Optimizer) -> float:
        new_lr = self.calc_lr()
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def update_momentum(self,
                        optimizer: torch.optim.Optimizer) -> float:
        new_momentum = self.calc_momentum()
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum
        return new_momentum

    def on_batch_begin(self, i: int, state: DotDict) -> None:
        if state.core.mode == "train":
            self.update_lr(state.core.opt)
            self.update_momentum(state.core.opt)


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

    def on_train_begin(self, state: DotDict) -> None:
        self.total_iter = self.len_loader * self.cycle_len - 1
        self.cut_point = int(self.total_iter * self.increase_fraction)

    def calc_lr(self) -> float:
        # calculate percent for learning rate change
        if self.cycle_iter <= self.cut_point:
            percent = self.cycle_iter / self.cut_point

        else:
            percent = 1 - (self.cycle_iter - self.cut_point) / (
                    self.total_iter - self.cut_point)

        res = self.init_lr * (1 + percent * (self.div_factor - 1)) / self.div_factor

        return res

    def calc_momentum(self) -> float:
        if self.cycle_iter <= self.cut_point:
            percent = 1 - self.cycle_iter / self.cut_point

        else:
            percent = (self.cycle_iter - self.cut_point) / (self.total_iter - self.cut_point)
        res = self.momentum_range[1] + percent * (
                self.momentum_range[0] - self.momentum_range[1])
        return res

    def on_batch_begin(self, i: int, state: DotDict) -> None:
        super().on_batch_begin(i, state)
        if state.core.mode == "train":
            self.cycle_iter += 1


class LRFinder(LRUpdater):
    """
    Helps you find an optimal learning rate for a model,
        as per suggetion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input.

    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self,
                 final_lr: float,
                 n_steps: int,
                 init_lr: float = 1e-6) -> None:
        super().__init__(init_lr)
        self.final_lr = final_lr
        self.n_steps = max(1, n_steps - 1)
        self.n = 0

    def calc_lr(self) -> float:
        res = self.init_lr * (self.final_lr / self.init_lr) ** \
              (self.n / self.n_steps)
        self.n += 1
        return res

    def calc_momentum(self) -> None:
        pass

    def update_momentum(self, optimizer: torch.optim.Optimizer) -> float:
        pass

    def on_batch_end(self, i: int, state: DotDict) -> None:
        super().on_batch_end(i, state)
        if self.n == (self.n_steps + 1):
            print("\nEnd of LRFinder")
            state.core.stop_epoch = True


class TBLogger(Callback):
    def __init__(self, logdir: Union[str, Path]) -> None:
        self.logdir = Path(logdir)
        self.writer = None
        self.total_iter = 0
        self.train_iter = 0
        self.val_iter = 0
        self.train_batch_iter = 0
        self.val_batch_iter = 0
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def update_total_iter(self, mode: str) -> None:
        if mode == "train":
            self.train_iter += 1
            self.train_batch_iter += 1
        if mode == "val":
            self.val_iter += 1
            self.val_batch_iter +=1
        self.total_iter += 1

    def on_train_begin(self, state: DotDict) -> None:
        self.train_iter = 0
        self.val_iter = 0
        self.logdir.mkdir(exist_ok=True)
        self.train_writer = SummaryWriter(str(self.logdir / "train"))
        self.val_writer = SummaryWriter(str(self.logdir / "val"))
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict):
        self.train_batch_iter = 0
        self.val_batch_iter = 0

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "train":
            for name, metric in state.core.metrics["train"].items():
                self.train_writer.add_scalar(f"batch/{name}",
                                             float(metric),
                                             global_step=self.total_iter)
                self.train_metrics[name].append(float(metric))

            lr = get_opt_lr(state.core.opt)
            self.train_writer.add_scalar("batch/lr",
                                         float(lr),
                                         global_step=self.train_iter)

            self.update_total_iter(state.core.mode)

        elif state.core.mode == "val":
            for name, metric in state.core.metrics["val"].items():
                self.val_writer.add_scalar(f"batch/{name}",
                                           float(metric),
                                           global_step=self.total_iter)
                self.val_metrics[name].append(float(metric))

            self.update_total_iter(state.core.mode)

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "train":
            for name, metric in self.train_metrics.items():
                mean = np.mean(metric[-self.train_batch_iter:])
                self.train_writer.add_scalar(f"epoch/{name}",
                                             float(mean),
                                             global_step=epoch)
        if state.core.mode == "val":
            for name, metric in self.val_metrics.items():
                mean = np.mean(metric[-self.val_batch_iter:])  # last epochs vals
                self.val_writer.add_scalar(f"epoch/{name}",
                                           float(mean),
                                           global_step=epoch)

    def on_train_end(self, state: DotDict) -> None:
        self.train_writer.close()
        self.val_writer.close()


class SimpleLossCallback(Callback):
    def __init__(self, target_key: str, preds_key: str) -> None:
        self.target_key = target_key
        self.preds_key = preds_key

    def on_batch_end(self, i: int, state: DotDict) -> None:
        target = state.core.batch[self.target_key]
        preds = state.core.out[self.preds_key]

        state.core.loss = state.core.criterion(preds, target)


class SimpleOptimizerCallback(Callback):
    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "train":
            state.core.opt.zero_grad()
            if state.core.use_fp16:
                with amp.scale_loss(state.core.loss, state.core.opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                state.core.loss.backward()
            state.core.opt.step()


class SimpleSchedulerCallback(Callback):
    def __init__(self,
                 sched: Union[_LRScheduler, ReduceLROnPlateau]) -> None:
        self.metric = "val_loss"
        if isinstance(sched, ReduceLROnPlateau):
            self.when = "on_epoch_end"
        else:
            self.when = "on_epoch_begin"

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        if self.when == "on_epoch_begin" and state.core.mode == "train":
            state.core.sched.step()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if self.when == "on_epoch_end" and state.core.mode == "train":
            state.core.sched.step(state.core.epoch_metrics[self.metric])


class ProgressBarCallback(Callback):
    def __init__(self) -> None:
        self.running_loss = None

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        self.running_loss = None

        loader = state.core.loader
        if state.core.mode == "train":
            description = f"Epoch {epoch+1}/{epochs}"
            state.core.pbar = get_pbar(loader, description)
        elif state.core.mode == "test":
            description = "Predict"
            state.core.pbar = get_pbar(loader, description)

    def on_batch_end(self, i: int, state: DotDict) -> None:

        if state.core.mode == "train":
            if not self.running_loss:
                self.running_loss = to_numpy(state.core.loss)

            self.running_loss = exp_weight_average(state.core.loss,
                                                   self.running_loss)
            postfix = {"loss": f"{self.running_loss:.4f}"}
            state.core.pbar.set_postfix(postfix)
            state.core.pbar.update()
        elif state.core.mode == "test":
            state.core.pbar.update()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "val":
            metrics = state.core.get("epoch_metrics", {})
            state.core.pbar.set_postfix_str(extend_postfix(state.core.pbar.postfix,
                                                      metrics))
            state.core.pbar.close()
        elif state.core.mode == "test":
            state.core.pbar.close()

class MetricsCallback(Callback):
    def __init__(self,
                 target_key: str,
                 preds_key: str,
                 metrics: Optional[Dict[str, Callable]] = None) -> None:
        self.metrics = metrics or {}
        self.pbar_metrics = None
        self.target_key = target_key
        self.preds_key = preds_key

    def get_metric(self,
                   metric: Callable,
                   target: torch.Tensor,
                   preds: Union[List, torch.Tensor]) -> float:
        # dataparallel workaround
        if isinstance(preds, list):
            preds = torch.cat(preds)
        return metric(target, preds)

    def update_epoch_metrics(self,
                             target: torch.Tensor,
                             preds: Union[List, torch.Tensor]) -> None:

        for name, m in self.metrics.items():
            value = self.get_metric(m, target, preds)
            self.pbar_metrics[name] += value

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        self.pbar_metrics = defaultdict(float)

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "val":
            self.pbar_metrics["val_loss"] += float(to_numpy(state.core.loss))
            self.update_epoch_metrics(target=state.core.batch[self.target_key],
                                      preds=state.core.out[self.preds_key])
        # tb logs
        if state.core.mode != "test" and state.core.do_log:
            state.core.metrics[state.core.mode]["loss"] = float(to_numpy(state.core.loss))
            for name, m in self.metrics.items():
                preds = state.core.out[self.preds_key]
                target = state.core.batch[self.target_key]
                value = self.get_metric(m, target, preds)
                state.core.metrics[state.core.mode][name] = value

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        divider = len(state.core.loader)
        for k in self.pbar_metrics.keys():
            self.pbar_metrics[k] /= divider
        state.core.epoch_metrics = self.pbar_metrics


class PredictionsSaverCallback(Callback):
    def __init__(self,
                 savepath: Optional[Union[str, Path]],
                 preds_key: str) -> None:
        if savepath is not None:
            self.savepath = Path(savepath)
            self.savepath.parent.mkdir(exist_ok=True)
            self.return_array = False
        else:
            self.savepath = None
            self.return_array = True
        self.preds_key = preds_key
        self.preds = []

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "test":
            out = state.core.out[self.preds_key]
            # DataParallelModel workaround
            if isinstance(out, list):
                out = np.concatenate([to_numpy(o) for o in out])
            else:
                out = to_numpy(out)
            self.preds.append(out)

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "test":
            preds = np.concatenate(self.preds)
            if self.return_array:
                state.core.preds = preds
            else:
                np.save(self.savepath, preds)
            self.preds = []


class CheckpointSaverCallback(Callback):
    def __init__(self,
                 savedir: str,
                 metric: Optional[str] = None,
                 n_best: int = 3,
                 prefix: Optional[str] = None,
                 mode: str = "min") -> None:
        self.metric = metric or "val_loss"
        self.n_best = n_best
        self.savedir = Path(savedir)
        self.prefix = f"{prefix}." if prefix is not None else "checkpoint."

        if mode not in ["min", "max"]:
            raise ValueError(f"mode should be 'min' or 'max', got {mode}")
        if mode == "min":
            self.maximize = False
        if mode == "max":
            self.maximize = True

        self.best_scores = []

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        # trim best scores
        self.best_scores = self.best_scores[:epochs]

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "val":
            score_val = state.core.epoch_metrics[self.metric]
            score_name = f"{self.prefix}{epoch + 1}.h5"
            score = (score_val, score_name)
            sorted_scores = sorted(self.best_scores + [score],
                                   reverse=self.maximize)
            self.best_scores = sorted_scores[:self.n_best]
            # set_trace()
            if score_name in (s[1] for s in self.best_scores):
                state.core.checkpoint = f"{self.savedir / score_name}"
                # remove worst checkpoint
                if len(sorted_scores) > self.n_best:
                    # set_trace()
                    Path(f"{self.savedir / sorted_scores[-1][1]}").unlink()

    def on_train_end(self, state: DotDict) -> None:
        best_cp = self.savedir / self.best_scores[0][1]
        shutil.copy(str(best_cp), f"{self.savedir}/{self.prefix}best.h5")
        print(f"\nCheckpoint\t{self.metric or 'val_loss'}")
        for score in self.best_scores:
            print(f"{self.savedir/score[1]}\t{score[0]:.6f}")


class EarlyStoppingCallback(Callback):
    def __init__(self,
                 patience: int,
                 metric: Optional[str] = None,
                 mode: str = "min",
                 min_delta: int = 0) -> None:
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

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "val":
            score = state.core.epoch_metrics[self.metric]
            if self.best_score is None:
                self.best_score = score
            if self.is_better(score, self.best_score):
                self.num_bad_epochs = 0
                self.best_score = score
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                state.core.stop_train = True


class DebuggerCallback(Callback):
    def __init__(self, when: List[str], modes: List[str]) -> None:
        self.when = when
        self.modes = modes

    def on_batch_begin(self, i: int, state: DotDict) -> None:
        if "on_batch_begin" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if "on_batch_end" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        if "on_epoch_begin" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if "on_epoch_end" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_train_begin(self, state: DotDict) -> None:
        if "on_train_begin" in self.when:
            set_trace()

    def on_train_end(self, state: DotDict) -> None:
        if "on_train_end" in self.when:
            set_trace()
