from typing import Type

import torch

from sklearn.metrics import accuracy_score


def accuracy(preds: Type[torch.Tensor],
             target: Type[torch.Tensor]) -> float:
    target = target.cpu().detach().numpy()
    # dataparallel workaround
    if isinstance(preds, list):
        preds = torch.cat(preds)
    preds = preds.cpu().detach().numpy().argmax(axis=1)
    return accuracy_score(target, preds)