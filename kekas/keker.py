from pdb import set_trace as st


from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple, Type, Dict, Union, Optional

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from .callbacks import Callback, Callbacks, ProgressBarCallback, \
    PredictionsSaverCallback, OneCycleLR, SimpleLossCallback, MetricsCallback, \
    TBLogger, LRFinder, CheckpointSaverCallback, SimpleSchedulerCallback, \
    EarlyStoppingCallback, SimpleOptimizerCallback
from .data import DataOwner
from .parallel import DataParallelCriterion, DataParallelModel
from .utils import DotDict, freeze_to, freeze, unfreeze


class Keker:
    """ The class serving the whole train-val-predict process.

    Args:
        model: The neural network for train/val/predict.
        dataowner: The namedtuple container of train/val/test dataloaders.
        tarket_key: The target/label key for batch-dict from dataloader.
            The dataloader returns batch as a dict on each iteration,
            that contains input data and target labels. This is key is for
            access to target labels in this dict.
        preds_key: The key for dict from self.step() functions.
            The self.step() function returns dict of predictions on each batch.
            This key is for access to predictions in this dict.
        criterion: The loss function (ex. : torch.nn.CrossEntropyLoss())
        metrics: {"name": metric_function} dict, that contains callable metrics
            for calculating. A metric takes target and predictions
            tensors as parameters, and returns float.
            For examples see kekas.metrics module.
        opt: pytorch Optimizer class (ex. torch.optim.SGD, torch.optm.Adam, etc)
            This optimizer will be used as default during training.
            Default optimizer is torch.optim.SGD.
        opt_params: The kwargs dict for optimizer initialization.
            It should contain any optimizer param you want EXCEPT learning rate,
            learing rate is specified in self.kek* methods.
        device: The torch.device when you want to put your model
        step_fn: The function that will be called at every batch of your data.
            Take a `self` as a parameter. In this function you define
            what you do with your batch. You get access to batch through
            self._state.batch object. Batch is a dict returned from your
            dataloader, and its key and values are specified in reader_function
            for DataKek.
            Return a dict that should contain batch predictions with access
            by `preds_key`.
            For example see self.default_step method and example ipynbs.
        loss_cb: The Callback for loss calculation. Define how to calculate loss
            and store loss value in self._state.loss attr.
            Default loss callback is SimpleLossCallback.
            For examples see kekas.callbacks.
        opt_cb: The Callback for optimizer applying.
            Default optimizer callback is SimpleOptimizerCallback.
            For examples see kekas.callbacks.
        callbacks: custom Callbacks thet will be applied before the core ones.
            For examples see example ipynbs.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 dataowner: DataOwner,
                 target_key: str = "label",
                 preds_key: str = "logits",
                 criterion: Optional[torch.nn.Module] = None,
                 metrics: Optional[Dict[str, Callable]] = None,
                 opt: Optional[Type[torch.optim.Optimizer]] = None,
                 opt_params: Optional[Dict] = None,
                 device: Optional[torch.device] = None,
                 step_fn: Optional[Callable] = None,
                 loss_cb: Optional[Callback] = None,
                 opt_cb: Optional[Callback] = None,
                 callbacks: Optional[Union[List, Callbacks]] = None) -> None:

        # The _state is an object that stores many variables and represents
        # the state of your train-val-repdict pipeline. _state passed to every
        # callback call.
        # You can use it as a container for your custom variables, but
        # DO NOT USE the following ones:
        #
        # loss, batch, model, dataowner, criterion, opt, parallel, checkpoint,
        # stop_iter, stop_epoch, stop_train, out, sched, mode, loader, pbar,
        # metrics, epoch_metrics
        #
        self._state = DotDict()

        self._state.model = model
        self._state.dataowner = dataowner

        self.target_key = target_key
        self.preds_key = preds_key

        self._state.criterion = criterion

        self._state.parallel = False
        if torch.cuda.device_count() > 1:
            self._state.model = DataParallelModel(self._state.model)
            self._state.criterion = DataParallelCriterion(self._state.criterion)
            self._state.parallel = True

        self.opt = opt or SGD
        self.opt_params = opt_params or {}
        self.device = device or torch.device("cuda" if
                                             torch.cuda.is_available()
                                             else "cpu")
        self._state.model.to(self.device)

        self.step = step_fn or self.default_step

        # the core callbacks for train-val-predict are determined here.
        # the order of callbacks is matters!
        loss_cb = loss_cb or SimpleLossCallback(target_key, preds_key)
        opt_cb = opt_cb or SimpleOptimizerCallback()
        metrics_cb = MetricsCallback(target_key, preds_key, metrics)

        callbacks = callbacks or []
        self.core_callbacks = callbacks + [loss_cb,
                                           metrics_cb,
                                           opt_cb,
                                           ProgressBarCallback()]
        callbacks = self.core_callbacks[:]

        self.callbacks = Callbacks(callbacks)

        self._state.checkpoint = ""

        # The number of batch in dataloader for iteration stop,
        # determined in self.kek* methods.
        self._state.stop_iter = None

        # flag for train stop after batch.
        self._state.stop_epoch = False

        # flag for stop the whole train, used for early stopping.
        self._state.stop_train = False

        # The scheduler attribute. Scheduler is determined in self.kek* methods.
        self._state.sched = None

    def kek(self,
            lr: float,
            epochs: int,
            skip_val: bool = False,
            opt: Optional[Type[torch.optim.Optimizer]] = None,
            opt_params: Optional[Dict] = None,
            sched: Optional[Callable] = None,
            sched_params: Optional[Dict] = None,
            sched_reduce_metric: Optional[str] = None,
            stop_iter: Optional[int] = None,
            logdir: Optional[Union[str, Path]] = None,
            cp_saver_params: Optional[Dict] = None,
            early_stop_params: Optional[Dict] = None) -> None:

        if stop_iter:
            self.stop_iter = stop_iter

        callbacks = self.callbacks

        opt = opt or self.opt
        opt_params = opt_params or self.opt_params
        params = (p for p in self._state.model.parameters() if p.requires_grad)
        self._state.opt = opt(params=params, lr=lr, **opt_params)

        if sched:
            sched_params = sched_params or {}
            self._state.sched = sched(optimizer=self._state.opt, **sched_params)
            sched_cb = SimpleSchedulerCallback(sched=self._state.sched,
                                               metric=sched_reduce_metric)
            self.callbacks = Callbacks(self.callbacks.callbacks + [sched_cb])

        if logdir:
            self._state.do_log = True
            self._state.metrics = defaultdict(dict)
            tboard_cb = TBLogger(logdir)
            self.callbacks = Callbacks(self.callbacks.callbacks + [tboard_cb])

        cp_saver_params = cp_saver_params or {}
        if cp_saver_params:
            cp_saver_cb = CheckpointSaverCallback(**cp_saver_params)
            self.callbacks = Callbacks(self.callbacks.callbacks + [cp_saver_cb])

        early_stop_params = early_stop_params or {}
        if early_stop_params:
            early_stop_cb = EarlyStoppingCallback(**early_stop_params)
            self.callbacks = Callbacks(self.callbacks.callbacks + [early_stop_cb])

        # try-finally to properly close progress bar
        try:
            self.callbacks.on_train_begin(self._state)

            for epoch in range(epochs):
                self.set_mode("train")
                self._run_epoch(epoch, epochs)

                if not skip_val:
                    self.set_mode("val")
                    self._run_epoch(epoch, epochs)

                if self._state.stop_train:
                    self._state.stop_train = False
                    print(f"Early stopped on {epoch + 1} epoch")
                    break

            self.callbacks.on_train_end(self._state)
        finally:
            self._state.pbar.close()
            self.callbacks = callbacks

    def kek_one_cycle(self,
                      max_lr: float,
                      cycle_len: int,
                      momentum_range: Tuple[float, float] = (0.95, 0.85),
                      div_factor: float = 25,
                      increase_fraction: float = 0.3,
                      opt: Optional[Type[torch.optim.Optimizer]] = None,
                      opt_params: Optional[Dict] = None,
                      logdir: Optional[Union[str, Path]] = None,
                      cp_saver_params: Optional[Dict] = None,
                      early_stop_params: Optional[Dict] = None) -> None:

        callbacks = self.callbacks

        # temporarily add OneCycle callback
        len_loader = len(self._state.dataowner.train_dl)
        one_cycle_cb = OneCycleLR(max_lr, cycle_len, len_loader,
                                  momentum_range, div_factor, increase_fraction)

        try:
            self.callbacks = Callbacks(callbacks.callbacks + [one_cycle_cb])

            self.kek(lr=max_lr,
                     epochs=cycle_len,
                     opt=opt,
                     opt_params=opt_params,
                     logdir=logdir,
                     cp_saver_params=cp_saver_params,
                     early_stop_params=early_stop_params)
        finally:
            # set old callbacks without OneCycle
            self.callbacks = callbacks

    def kek_lr(self,
               final_lr: float,
               logdir: Union[str, Path],
               init_lr: float = 1e-6,
               n_steps: Optional[int] = None) -> None:

        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True)
        tmp_cp = logdir / "tmp.h5"
        self.save(tmp_cp)

        n_steps = n_steps or len(self._state.dataowner.train_dl)

        callbacks = self.callbacks

        try:
            lrfinder_cb = LRFinder(final_lr=final_lr,
                                   init_lr=init_lr,
                                   n_steps=n_steps)

            self.callbacks = Callbacks(self.core_callbacks + [lrfinder_cb])
            self.kek(lr=init_lr, epochs=1, skip_val=True, logdir=logdir)
        finally:
            self.callbacks = callbacks
            self.load(tmp_cp)
            tmp_cp.unlink()

    def _run_epoch(self,
                   epoch: int,
                   epochs: int) -> None:
        self.callbacks.on_epoch_begin(epoch, epochs, self._state)

        with torch.set_grad_enabled(self.is_train):
            for i, batch in enumerate(self._state.loader):
                self.callbacks.on_batch_begin(i, self._state)

                self._state.batch = self.to_device(batch)

                self._state.out = self.step()

                self.callbacks.on_batch_end(i, self._state)

                if (self._state.stop_iter and self._state.mode == "train"
                        and i == self._state.stop_iter - 1):
                    # break only in train mode and if early stop is set
                    self._state.stop_epoch = True

                if self._state.stop_epoch:
                    self._state.stop_epoch = False
                    # st()
                    break

        self.callbacks.on_epoch_end(epoch, self._state)

        if self._state.checkpoint:
            self.save(self._state.checkpoint)
            self._state.checkpoint = ""

    def default_step(self) -> Dict[str, torch.Tensor]:
        inp = self._state.batch["image"]
        logits = self._state.model(inp)

        return {"logits": logits}

    def predict(self) -> None:
        self.set_mode("test")
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

    def predict_loader(self,
                       loader: Type[DataLoader],
                       savepath: Union[str, Path]) -> None:
        callbacks = self.callbacks

        tmp_callbacks = Callbacks([ProgressBarCallback(),
                                   PredictionsSaverCallback(savepath,
                                                            self.preds_key)])

        self.callbacks = tmp_callbacks

        self._state.mode = "test"
        self._state.loader = loader
        self._state.model.eval()
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

        self.callbacks = callbacks

    def predict_tensor(self,
                       tensor: Type[torch.Tensor],
                       to_numpy: bool = False) -> Union[Type[torch.Tensor],
                                                        np.ndarray]:
        tensor = tensor.to(self.device)
        with torch.set_grad_enabled(False):
            self.set_mode("test")
            preds = self._state.model(tensor)
        if to_numpy:
            preds = preds.cpu().numpy()
        return preds

    def predict_array(self,
                      array: np.ndarray,
                      to_numpy: bool = False) -> Union[Type[torch.Tensor],
                                                       np.ndarray]:
        tensor = torch.from_numpy(array)
        return self.predict_tensor(tensor, to_numpy)

    def TTA(self,
            loader: Type[DataLoader],
            tfms: Union[List, Dict],
            savedir: Union[str, Path],
            prefix: str = "preds") -> None:

        if isinstance(tfms, dict):
            names = [f"{prefix}_{k}.npy" for k in tfms]
            tfms = tfms.values()
        elif isinstance(tfms, list):
            names = [f"{prefix}_{i}.npy" for i in range(len(tfms))]
        else:
            raise ValueError(f"Transforms should be List or Dict, got {type(tfms)}")

        default_tfms = loader.dataset.transforms
        for name, tfm in zip(names, tfms):
            loader.dataset.transforms = tfm
            savepath = Path(savedir) / name
            self.predict_loader(loader, savepath)
        loader.dataset.transforms = default_tfms

    def save(self, savepath: Union[str, Path]) -> None:
        savepath = Path(savepath)
        savepath.parent.mkdir(exist_ok=True)
        torch.save(self._state.model.state_dict(), savepath)

    def load(self, loadpath: Union[str, Path]) -> None:
        loadpath = Path(loadpath)
        checkpoint = torch.load(loadpath,
                                map_location=lambda storage, loc: storage)
        if not isinstance(self._state.model, DataParallelModel) \
                and "module." in list(checkpoint.keys())[0]:
            # [7:] is to skip 'module.' in group name
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        self._state.model.load_state_dict(checkpoint)

    def to_device(self,
                  batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()
                if hasattr(v, "to")}

    def set_mode(self, mode: str) -> None:
        if mode == "train":
            self._state.model.train()
            self._state.loader = self._state.dataowner.train_dl
        elif mode == "val":
            self._state.model.eval()
            self._state.loader = self._state.dataowner.val_dl
        elif mode == "test":
            self._state.model.eval()
            self._state.loader = self._state.dataowner.test_dl
        self._state.mode = mode

    def freeze_to(self, n: int,
                  freeze_bn: bool = False,
                  model_attr: Optional[str] = None) -> None:

        module = self.get_model_attr(model_attr)
        freeze_to(module, n, freeze_bn)

    def freeze(self,
               freeze_bn: bool = False,
               model_attr: Optional[str] = None) -> None:
        module = self.get_model_attr(model_attr)
        freeze(module, freeze_bn)

    def unfreeze(self,
                 model_attr: Optional[str] = None) -> None:
        module = self.get_model_attr(model_attr)
        unfreeze(module)

    def get_model_attr(self, model_attr: str):
        if self._state.parallel:
            model = self._state.model.module
        else:
            model = self._state.model

        if model_attr is not None:
            module = getattr(model, model_attr)
        else:
            module = model
        return module

    @property
    def is_train(self) -> bool:
        return self._state.mode == "train"
