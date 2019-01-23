from functools import reduce

import sys

from tqdm import tqdm


def to_numpy(data):
    return data.detach().cpu().numpy()


def exp_weight_average(curr_val, prev_val, alpha=0.9, from_torch=True):
    if from_torch:
        curr_val = to_numpy(curr_val)
    return alpha * prev_val + (1 - alpha) * curr_val


def get_pbar(dataloader, description):

    pbar = tqdm(
        total=len(dataloader),
        leave=True,
        ncols=0,
        desc=description,
        file=sys.stdout)

    return pbar


def extend_postfix(postfix, dct):
    postfixes = [postfix] + [f"{k}={v:.4f}" for k, v in dct.items()]
    return ", ".join(postfixes)

# TODO: REMOVE
def update_epoch_metrics(target, preds, metrics, epoch_metrics):
    for m in metrics:
        value = m(target, preds)
        epoch_metrics[m.__name__] += value


def get_opt_lr(opt):
    # TODO: rewrite it for differentrial learning rates
    lrs = [pg["lr"] for pg in opt.param_groups]
    res = reduce(sum, lrs) / len(lrs)
    return res


class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]
