class State:
    def __init__(self, inp=None, out=None, target=None, loss=None, opt=None,
                 is_train=None):
        self.inp = inp
        self.out = out
        self.target = target
        self.loss = loss
        self.opt = opt
        self.is_train = is_train

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
