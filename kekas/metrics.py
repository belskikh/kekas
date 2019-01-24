from pdb import set_trace as st

from sklearn.metrics import accuracy_score


def accuracy(target, preds):
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy().argmax(axis=1)
    return accuracy_score(target, preds)