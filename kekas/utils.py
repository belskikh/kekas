from pdb import set_trace as st

from functools import reduce
from pathlib import Path
import os
import sys
from typing import Any, Dict, Union, Hashable, Optional, List

import numpy as np
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator, ScalarEvent
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import logging
logging.getLogger('tensorflow').addFilter(lambda x: 0)

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def freeze_to(module: nn.Module,
              n: int,
              freeze_bn: bool = False) -> None:
    layers = list(module.children())
    for l in layers[:n]:
        for module in flatten_layer(l):
            if freeze_bn or not isinstance(module, BN_TYPES):
                set_grad(module, requires_grad=False)

    for l in layers[n:]:
        for module in flatten_layer(l):
            set_grad(module, requires_grad=True)


def freeze(module: nn.Module,
           freeze_bn: bool = False) -> None:
    freeze_to(module=module, n=-1, freeze_bn=freeze_bn)


def unfreeze(module: nn.Module) -> None:
    layers = list(module.children())
    for l in layers:
        for module in flatten_layer(l):
            set_grad(module, requires_grad=True)


def set_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


# https://github.com/fastai/fastai/blob/6778fd518e95ea8e1ce1e31a2f96590ee254542c/fastai/torch_core.py#L157
class ParameterModule(nn.Module):
    """Register a lone parameter `p` in a module."""
    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x): return x


# https://github.com/fastai/fastai/blob/6778fd518e95ea8e1ce1e31a2f96590ee254542c/fastai/torch_core.py#L149
def children_and_parameters(m: nn.Module):
    """Return the children of `m` and its direct parameters not registered in modules."""
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],
                     [])
    for p in m.parameters():
        if id(p) not in children_p:
            st()
            children.append(ParameterModule(p))
    return children


def flatten_layer(layer: nn.Module) -> List[nn.Module]:
    if len(list(layer.children())):
        layers = []
        for children in children_and_parameters(layer):
            layers += flatten_layer(children)
        return layers
    else:
        return [layer]


def to_numpy(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy()


def exp_weight_average(curr_val: Union[float, torch.Tensor],
                       prev_val: float,
                       alpha: float = 0.9) -> float:
    if isinstance(curr_val, torch.Tensor):
        curr_val = to_numpy(curr_val)
    return float(alpha * prev_val + (1 - alpha) * curr_val)


def get_pbar(dataloader: DataLoader,
             description: str) -> tqdm:

    pbar = tqdm(
        total=len(dataloader),
        leave=True,
        ncols=0,
        desc=description,
        file=sys.stdout)

    return pbar


def extend_postfix(postfix: str, dct: Dict) -> str:
    if postfix is None:
        postfix = ""
    postfixes = [postfix] + [f"{k}={v:.4f}" for k, v in dct.items()]
    return ", ".join(postfixes)


def get_opt_lr(opt: torch.optim.Optimizer) -> float:
    lrs = [pg["lr"] for pg in opt.param_groups]
    res = reduce(lambda x, y: x + y, lrs) / len(lrs)
    return res


class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr: str) -> Any:
        return self.get(attr)

    def __setattr__(self, key: Hashable, value: Any) -> Any:
        self.__setitem__(key, value)

    def __setitem__(self, key: Hashable, value: Any) -> Any:
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item: str) -> None:
        self.__delitem__(item)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self.__dict__[key]


def load_state_dict(model: torch.nn.Module,
                    state_dict: Dict,
                    skip_wrong_shape: bool = False):
    model_state_dict = model.state_dict()

    for key in state_dict:
        if key in model_state_dict:
            if model_state_dict[key].shape == state_dict[key].shape:
                model_state_dict[key] = state_dict[key]
            elif not skip_wrong_shape:
                m = f"Shapes of the '{key}' parameters do not match: " \
                    f"{model_state_dict[key].shape} vs {state_dict[key].shape}"
                raise Exception(m)

    model.load_state_dict(model_state_dict)


def get_tensorboard_scalars(logdir: str,
                            metrics: Optional[List[str]],
                            step: str) -> Dict[str, List]:
    event_acc = EventAccumulator(str(logdir))
    event_acc.Reload()

    if metrics is not None:
        scalar_names = [n for n in event_acc.Tags()["scalars"] if step in n
                        and any(m in n for m in metrics)]
    else:
        scalar_names = [n for n in event_acc.Tags()["scalars"] if step in n]

    scalars = {sn: event_acc.Scalars(sn) for sn in scalar_names}
    return scalars


def get_scatter(scalars: Dict[str, ScalarEvent],
                name: str,
                prefix: str) -> go.Scatter:
    xs = [s.step for s in scalars[name]]
    ys = [s.value for s in scalars[name]]

    return go.Scatter(x=xs, y=ys, name=prefix+name)


def plot_tensorboard_log(logdir: Union[str, Path],
                         step: Optional[str] = "batch",
                         metrics: Optional[List[str]] = None,
                         height: Optional[int] = None,
                         width: Optional[int] = None) -> None:
    init_notebook_mode(connected=True)
    logdir = Path(logdir)

    train_scalars = get_tensorboard_scalars(logdir/"train", metrics, step)
    val_scalars = get_tensorboard_scalars(logdir/"val", metrics, step)

    if height is not None:
        height = height // len(train_scalars)

    for m in train_scalars:
        tm = get_scatter(train_scalars, m, prefix="train/")
        try:
            vm = get_scatter(val_scalars, m, prefix="val/")
            data = [tm, vm]
        except:
            data = [tm]
        layout = go.Layout(title=m,
                           height=height,
                           width=width,
                           yaxis=dict(hoverformat=".6f"))
        iplot(go.Figure(data=data, layout=layout))
