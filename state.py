from .utils import DotDict


class State(DotDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opt = None
        self.mode = None
        self.batch_report = {}
        self.epoch_report = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def flash_batch_report(self):
        self.batch_report = {}

    def flash_epoch_report(self):
        self.epoch_report = {}
