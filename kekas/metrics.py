from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from typing import Type


def accuracy(preds: torch.Tensor,
             target: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy().argmax(axis=1)
    return accuracy_score(target, preds)


def bce_accuracy(preds: torch.Tensor,
                 target: torch.Tensor,
                 thresh: bool = 0.5) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)
  

def roc_auc(preds: torch.Tensor,
            target: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)
