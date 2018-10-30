class State:
    __slots__ = ("inp", "out", "target", "loss")

    def __init__(self, inp=None, out=None, target=None, loss=None):
        self.inp = inp
        self.out = out
        self.target = target
        self.loss = loss

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
