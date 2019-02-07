from functools import reduce
import sys
from typing import Any, Dict, Union, Hashable

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

def to_numpy(data: Type[torch.Tensor]) -> np.ndarray:
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
