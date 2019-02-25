from typing import Type

import torch

from sklearn.metrics import accuracy_score


def accuracy(target: torch.Tensor,
             preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy().argmax(axis=1)
    return accuracy_score(target, preds)