from pdb import set_trace as st

import torch

from sklearn.metrics import accuracy_score


def accuracy(target, preds):
    target = target.cpu().detach().numpy()
    # dataparallel workaround
    if isinstance(preds, list):
        preds = torch.cat(preds)
    preds = preds.cpu().detach().numpy().argmax(axis=1)
    return accuracy_score(target, preds)