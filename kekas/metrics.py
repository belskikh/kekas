from typing import Type

import torch

from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(target: torch.Tensor,
             preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy().argmax(axis=1)
    return accuracy_score(target, preds)

def bce_accuracy(target: torch.Tensor,
                 preds: torch.Tensor,
                 thresh: bool = 0.5) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)
  
def roc_auc(target: torch.Tensor,
                 preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)
