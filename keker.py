from tqdm import tqdm, tnrange

import torch
from torch.optim import Adam

class Keker():
    ''' The core class that proivdes main methods for training and predicting on given model and dataset
    '''
    def __init__(self, model, train_dl, val_dl=None, test_dl=None,
                 optimizer=None, criterion=None, metrics=None):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.optimizer = optimizer or Adam(1e-3)
        self.criterion = criterion
        self.metrics = metrics or []


    def kek(self, lr, epochs, one_cycle=(10, 2, 0.95, 0.85), w_decay=10e-5,
            sampler=None, stepper=None, callbacks=None):
        if callbacks is None:
            callbacks = []
        batch_num = 0
        avg_loss = 0.

        for epoch in tnrange(epochs, desc='Epoch'):
            dl_iter = tqdm(iter(self.train_dl), leave=False, total=len(self.train_dl), miniters=0)

            for (*inp, target) in dl_iter:
                batch_num += 1
                for cb in callbacks:
                    cb.on_batch_begin()

                loss = stepper.step(*inp, target)

                for cb in callbacks:
                    cb.on_batch_end()


    def lr_range_test(self, start_lr=1e-5, stop_lr=1e-2, n_iter=None, w_decay=None, linear=False,):
        pass

    def predict(self, dl=None):
        dl = dl or self.test_dl
        pass

    def TTA(self, n_aug=6):
        pass

    def save(self, path):
        pass

    def load(self, weights):
        pass


class Stepper():
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # self.reset(True)

    # def reset(self, train=True):
    #     if train: apply_leaf(self.m, set_train_mode)
    #     else: self.m.eval()
    #     if hasattr(self.m, 'reset'):
    #         self.model.reset()


    def step(self, inp, target):

        output = self.model(*inp)


    # def evaluate(self, xs, y):
    #     preds = self.m(*xs)
    #     if isinstance(preds,tuple): preds=preds[0]
    #     return preds, self.crit(preds, y)