from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Tuple, Type, Dict, Union, Optional
import warnings

try:
    from apex import amp
except ImportError as e:
    warnings.warn(f"Error '{e}'' during importing apex library. To use mixed precison"
                  " you should install it from https://github.com/NVIDIA/apex")

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn.parallel import DistributedDataParallel

from .callbacks import Callback, Callbacks, ProgressBarCallback, \
    PredictionsSaverCallback, OneCycleLR, SimpleLossCallback, MetricsCallback, \
    TBLogger, LRFinder, CheckpointSaverCallback, SimpleSchedulerCallback, \
    EarlyStoppingCallback, SimpleOptimizerCallback
from .data import DataOwner
from .parallel import DataParallelCriterion, DataParallelModel
from .utils import DotDict, freeze_to, freeze, unfreeze, load_state_dict, \
    plot_tensorboard_log


class Keker:
    """ The class serving the whole train-val-predict process.

    Args:
        model: The neural network for train/val/predict.
        dataowner: The namedtuple container of train/val/test dataloaders.
        criterion: The loss function or the dict {'name': loss function}
            in case of multiple loss setup. If multiple loss is using,
            loss_cb should be provided.
            (ex. : torch.nn.CrossEntropyLoss(),
            {"ce": torch.nn.CrossEntropyLoss(), "bce": torch.nn.BCE()})
        tarket_key: The target/label key for batch-dict from dataloader.
            The dataloader returns batch as a dict on each iteration,
            that contains input data and target labels. This is key is for
            access to target labels in this dict.
        preds_key: The key for dict from self.step() functions.
            The self.step() function returns dict of predictions on each batch.
            This key is for access to predictions in this dict.
            This attribute is optional in default behavior but coulb be used
            in case of using custom callbacks.
        metrics: {"name": metric_function} dict, that contains callable metrics
            for calculating. The metric takes prediction and target
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
            self._state.core.batch object. Batch is a dict returned from your
            dataloader, and its key and values are specified in reader_function
            for DataKek.
            Return a dict that should contain batch predictions with access
            by `preds_key`.
            For example see self.default_step method and example ipynbs.
        loss_cb: The Callback for loss calculation. Define how to calculate loss
            and store loss value in self._state.core.loss attr.
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
                 dataowner: Optional[DataOwner] = None,
                 criterion: Optional[Union[torch.nn.Module,
                                           Dict[str, torch.nn.Module]]] = None,
                 target_key: str = "label",
                 preds_key: str = "preds",
                 metrics: Optional[Dict[str, Callable]] = None,
                 opt: Optional[Type[torch.optim.Optimizer]] = None,
                 opt_params: Optional[Dict] = None,
                 device: Optional[torch.device] = None,
                 step_fn: Optional[Type[Callback]] = None,
                 loss_cb: Optional[Type[Callback]] = None,
                 opt_cb: Optional[Type[Callback]] = None,
                 callbacks: Optional[Union[List, Callbacks]] = None) -> None:

        # The state is an object that stores many variables and represents
        # the state of your train-val-predict pipeline. state passed to every
        # callback call.
        # You can use it as a container for your custom variables, but
        # DO NOT USE the 'core' attrubute, because all kekers variables are
        # there
        #
        self.state = DotDict()
        self.state.core = DotDict()

        self.state.core.model = model
        self.state.core.dataowner = dataowner or DataOwner(None, None, None)

        self.target_key = target_key
        self.preds_key = preds_key

        self.state.core.criterion = criterion

        self.state.core.parallel = False
        if torch.cuda.device_count() > 1:
            self.state.core.model = DataParallelModel(self.state.core.model)
            if self.state.core.criterion is not None:
                if isinstance(self.state.core.criterion, dict):
                    self.state.core.criterion = {
                        k: DataParallelCriterion(v) for k, v
                        in self.state.core.criterion.items()
                    }
                else:
                    self.state.core.criterion = DataParallelCriterion(
                        self.state.core.criterion
                    )
            self.state.core.parallel = True

        self.opt = opt or SGD
        self.opt_params = opt_params or {}
        self.device = device or torch.device("cuda" if
                                             torch.cuda.is_available()
                                             else "cpu")
        self.state.core.model.to(self.device)

        self.step_fn = step_fn or self.default_step_fn

        # the core callbacks for train-val-predict are determined here.
        # the order of callbacks is matters!
        if loss_cb is not None:
            loss_cb = loss_cb(target_key, self.preds_key)
        else:
            loss_cb = SimpleLossCallback(target_key, self.preds_key)

        opt_cb = opt_cb or SimpleOptimizerCallback()
        metrics_cb = MetricsCallback(target_key, self.preds_key, metrics)

        callbacks = callbacks or []
        self.core_callbacks = callbacks + [loss_cb,
                                           metrics_cb,
                                           opt_cb,
                                           ProgressBarCallback()]
        callbacks = self.core_callbacks[:]

        self.callbacks = Callbacks(callbacks)

        self.state.core.checkpoint = ""

        # The number of batch in dataloader for iteration stop,
        # determined in self.kek* methods.
        self.state.core.stop_iter = None

        # flag for train stop after batch.
        self.state.core.stop_epoch = False

        # flag for stop the whole train, used for early stopping.
        self.state.core.stop_train = False

        # The scheduler attribute. Scheduler is determined in self.kek* methods.
        self.state.core.sched = None

        # Flag for logger callback
        self.state.core.do_log = False

        # Predictions variable
        self.state.core.preds = None

        # FP16 flag
        self.state.core.use_fp16 = False

    def kek(self,
            lr: float,
            epochs: int,
            skip_val: bool = False,
            opt: Optional[Type[torch.optim.Optimizer]] = None,
            opt_params: Optional[Dict] = None,
            sched: Optional[Callable] = None,
            sched_params: Optional[Dict] = None,
            stop_iter: Optional[int] = None,
            logdir: Optional[Union[str, Path]] = None,
            cp_saver_params: Optional[Dict] = None,
            early_stop_params: Optional[Dict] = None) -> None:
        """Kek your model to the moon!

        Conducts a standard train-val procedure with several options for
        customization.

        Args:
            lr: learining rate
            epochs: number of epochs to train
            opt: torch optimizer. If specified, than specified optimizer will be
                used for this train-val procedure, else the default one.
            opt_params: The kwargs dict for custom optimizer initialization.
                It should contain any opt params you want EXCEPT learning rate,
                Can be defined even for default optimizer.
            sched: optional pytorch scheduler class. If specified, sched_params
                must be specified too.
                Ex: torch.optim.lr_scheduler.StepLR.
            sched_params: kwargs dict parameters for scheduler
            stop_iter: number of batch when you want to end an epoch
            logdir: If provided, the TBLogger will be created and tensorboard
                logs will be written in this directory.
                For more info see kekas.callbacks.TBLogger and example ipynb's.
            cp_saver_params: kwargs dict parameters for CheckpointSaverCallback.
                If provided, then a CheckpointSaverCallback will be created.
                For more info see kekas.callbacks.CheckpointSaverCallback
                and example ipynb's.
            early_stop_params: kwargs dict parameters for EarlyStoppingCallback.
                If provided, then a EarlyStoppingCallback will be created.
                For more info see kekas.callbacks.EarlyStoppingCallback
                and example ipynb's.
        """

        # check if criterion exists
        if self.state.core.criterion is None:
            raise Exception("Keker needs criterion. "
                            "Reinitialize Keker with one.")

        # check if dataowner exists
        if self.state.core.dataowner.train_dl is None:
            raise Exception("Keker needs Dataowner. "
                            "Reinitialize Keker with one.")

        if stop_iter:
            self.stop_iter = stop_iter

        # save callbacks
        callbacks = self.callbacks

        opt = opt or self.opt
        opt_params = opt_params or self.opt_params
        params = (p for p in self.state.core.model.parameters() if p.requires_grad)
        self.state.core.opt = opt(params=params, lr=lr, **opt_params)

        if self.state.core.use_fp16:
            _, self.state.core.opt = amp.initialize(self.state.core.model,
                                                    self.state.core.opt,
                                                    opt_level="O1",
                                                    verbosity=0)

        if sched:
            sched_params = sched_params or {}
            self.state.core.sched = sched(optimizer=self.state.core.opt, **sched_params)
            sched_cb = SimpleSchedulerCallback(sched=self.state.core.sched)
            self.callbacks = Callbacks(self.callbacks.callbacks + [sched_cb])

        if logdir:
            self.state.core.do_log = True
            self.state.core.metrics = defaultdict(dict)
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

        # try-finally to properly close progress bar and restore callbacks
        try:
            self.callbacks.on_train_begin(self.state)

            for epoch in range(epochs):
                self.set_mode("train")
                self._run_epoch(epoch, epochs)

                if not skip_val:
                    self.set_mode("val")
                    self._run_epoch(epoch, epochs)

                if self.state.core.stop_train:
                    self.state.core.stop_train = False
                    print(f"Early stopped on {epoch + 1} epoch")
                    break

            self.callbacks.on_train_end(self.state)
        finally:
            self.state.core.pbar.close()
            self.state.core.do_log = False
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
        """Kek your model to the moon with One Cycle policy!

        Conducts a one-cycle train-val procedure with several options for
        customization.
        For info about One Cycle policy please see:
        https://arxiv.org/abs/1803.09820
        https://sgugger.github.io/the-1cycle-policy.html

        Args:
            max_lr: the maximum learning rate that will be achieved during
                training process
            cycle_len: the number of full passes through the training dataset.
                It is quite similar to epochs number
            momentum_range: the range of optimizers momentum changes
            div_factor: is equal to max_lr / min_lr during one-cycle training
            increase_fraction: the fraction of the whole iterations during which
                the learning rate will increase
            opt: torch optimizer. If specified, than specified optimizer will be
                used for this train-val procedure, else the default one.
            opt_params: The kwargs dict for custom optimizer initialization.
                It should contain any opt params you want EXCEPT learning rate,
                Can be defined even for default optimizer.
            logdir: If provided, the TBLogger will be created and tensorboard
                logs will be written in this directory.
                For more info see kekas.callbacks.TBLogger and example ipynb's.
            cp_saver_params: kwargs dict parameters for CheckpointSaverCallback.
                If provided, then a CheckpointSaverCallback will be created.
                For more info see kekas.callbacks.CheckpointSaverCallback
                and example ipynb's.
            early_stop_params: kwargs dict parameters for EarlyStoppingCallback.
                If provided, then a EarlyStoppingCallback will be created.
                For more info see kekas.callbacks.EarlyStoppingCallback
                and example ipynb's.
        """

        callbacks = self.callbacks

        # temporarily add OneCycle callback
        len_loader = len(self.state.core.dataowner.train_dl)
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
               n_steps: Optional[int] = None,
               opt: Optional[Type[torch.optim.Optimizer]] = None,
               opt_params: Optional[Dict] = None) -> None:
        """Help you kek your model to the moon by finding "optimal" lr!

        Conducts the learning rate find procedure.
        For info please see:
        https://arxiv.org/abs/1803.09820
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

        Args:
            final_lr: the learning rate at the end of the lr find process
            logdir: the directory for tensorboard logs, that will allow you
                analyze the loss dynamic.
            init_lr: the learning rate at the start of the lr find process
            n_steps: the number of iterations of lr find process. If provided
                finding process will stop at this iteration, else lr find
                process will last one epoch
            opt: torch optimizer. If specified, than specified optimizer will be
                used for this train-val procedure, else the default one.
            opt_params: The kwargs dict for custom optimizer initialization.
                It should contain any opt params you want EXCEPT learning rate,
                Can be defined even for default optimizer.
        """

        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True)
        tmp_cp = logdir / "tmp.h5"
        self.save(tmp_cp)

        len_loader = len(self.state.core.dataowner.train_dl)
        n_steps = n_steps if n_steps is not None else len_loader
        n_epochs = max(1, int(np.ceil(n_steps / len_loader)))

        callbacks = self.callbacks

        try:
            lrfinder_cb = LRFinder(final_lr=final_lr,
                                   init_lr=init_lr,
                                   n_steps=n_steps)

            self.callbacks = Callbacks(self.core_callbacks + [lrfinder_cb])
            self.kek(lr=init_lr, epochs=n_epochs, skip_val=True, logdir=logdir,
                     opt=opt, opt_params=opt_params)
        finally:
            self.callbacks = callbacks
            self.load(tmp_cp)
            tmp_cp.unlink()

    def _run_epoch(self,
                   epoch: int,
                   epochs: int) -> None:
        """Run one epoch of train-val procedure

        Args:
            epoch: number of the current epoch
            epochs: total number of epochs
        """
        self.callbacks.on_epoch_begin(epoch, epochs, self.state)

        with torch.set_grad_enabled(self.is_train):
            for i, batch in enumerate(self.state.core.loader):
                self.callbacks.on_batch_begin(i, self.state)

                self.state.core.batch = self.to_device(batch)

                self.state.core.out = self.step()

                self.callbacks.on_batch_end(i, self.state)

                if (self.state.core.stop_iter and self.state.core.mode == "train"
                        and i == self.state.core.stop_iter - 1):
                    # break only in train mode and if early stop is set
                    self.state.core.stop_epoch = True

                if self.state.core.stop_epoch:
                    self.state.core.stop_epoch = False
                    # st()
                    break

        self.callbacks.on_epoch_end(epoch, self.state)

        if self.state.core.checkpoint:
            self.save(self.state.core.checkpoint)
            self.state.core.checkpoint = ""

    @staticmethod
    def default_step_fn(model: torch.nn.Module,
                        batch: torch.Tensor) -> torch.Tensor:
        """Determine what your model will do with your data.

        Args:
            model: the pytorch module to pass input in
            batch: the batch of data from the DataLoader

        Returns:
            The models forward pass results
        """
        inp = batch["image"]
        return model(inp)

    def step(self) -> Dict[str, torch.Tensor]:
        """The step function that calls each iteration.
        Wraps the self.step_fn to provide a dict of predictions

        Returns:
            the dict that contains prediction tensor for the batch.
        """
        preds = self.step_fn(model=self.state.core.model,
                             batch=self.state.core.batch)

        return {self.preds_key: preds}

    def predict(self,
                savepath: Optional[Union[str, Path]] = None
                ) -> Union[None, np.ndarray]:
        """Infer the model on test dataloader and saves prediction as numpy array

        Args:
            savepath: the directory to save predictions. If not passed,
                the method will return a numpy array of predictions.
        """
        return self.predict_loader(loader=self.state.core.dataowner.test_dl,
                                   savepath=savepath)

    def predict_loader(self,
                       loader: DataLoader,
                       savepath: Optional[Union[str, Path]] = None
                       ) -> Union[None, np.ndarray]:
        """Infer the model on dataloader and saves prediction as numpy array

        Args:
            loader: the dataloader for generating predictions
            savepath: the directory to save predictions. If not passed,
                the method will return a numpy array of predictions.
        """
        callbacks = self.callbacks

        tmp_callbacks = Callbacks([ProgressBarCallback(),
                                   PredictionsSaverCallback(savepath,
                                                            self.preds_key)])

        self.callbacks = tmp_callbacks

        self.state.core.mode = "test"
        self.state.core.loader = loader
        self.state.core.model.eval()
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

        self.callbacks = callbacks

        preds = self.state.core.preds
        self.state.core.preds = None
        return preds

    def predict_tensor(self,
                       tensor: Type[torch.Tensor],
                       to_numpy: bool = False) -> Union[Type[torch.Tensor],
                                                        np.ndarray]:
        """Infer the model on one torch Tensor.

        Args:
            tensor: torch tensor to predict on.
                Should has [batch_size, *(one_sample_shape)] shape
            to_numpy: if True, converts predictions to numpy array

        Returns:
            Predictions on input tensor.
        """
        tensor = tensor.to(self.device)
        with torch.set_grad_enabled(False):
            self.set_mode("test")
            preds = self.state.core.model(tensor)

            # dataparallel workaround
            if isinstance(preds, list):
                preds = torch.cat(preds)
        if to_numpy:
            preds = preds.cpu().numpy()
        return preds

    def predict_array(self,
                      array: np.ndarray,
                      to_numpy: bool = False) -> Union[Type[torch.Tensor],
                                                       np.ndarray]:
        """Infer the model on one numpy array.

        Args:
            array: numpy array to predict on.
                Should has [batch_size, *(one_sample_shape)] shape
            to_numpy: if True, converts predictions to numpy array

        Returns:
            Predictions on input tensor.
        """
        tensor = torch.from_numpy(array)
        return self.predict_tensor(tensor, to_numpy)

    def TTA(self,
            loader: DataLoader,
            tfms: Union[List, Dict],
            savedir: Union[str, Path],
            prefix: str = "preds") -> None:
        """Conduct the test-time augmentations procedure.

        Create predictions for each set of provided transformations and saves
        each prediction in savedir as a numpy arrays.

        Args:
            loader: loader to predict
            tfms: the list with torchvision.transforms or
                  the dict with {"name": torchvision.transforms} pairs.
                  List indexes or dict keys will be used for generating
                  predictions names.
            savedir: the directory to save predictions
            prefix: the prefix for predictions files names
        """
        if isinstance(tfms, dict):
            names = [f"{prefix}_{k}.npy" for k in tfms]
            tfms = tfms.values()
        elif isinstance(tfms, list):
            names = [f"{prefix}_{i}.npy" for i in range(len(tfms))]
        else:
            raise ValueError(f"Transforms should be List or Dict, "
                             f"got {type(tfms)}")

        default_tfms = loader.dataset.transforms
        for name, tfm in zip(names, tfms):
            loader.dataset.transforms = tfm
            savepath = Path(savedir) / name
            self.predict_loader(loader, savepath)
        loader.dataset.transforms = default_tfms

    def save(self, savepath: Union[str, Path]) -> None:
        """Save models state dict on the specified path.

        Args:
            savepath: the path to save the state dict.
        """
        savepath = Path(savepath)
        savepath.parent.mkdir(exist_ok=True)
        torch.save(self.state.core.model.state_dict(), savepath)

    def load(self,
             loadpath: Union[str, Path],
             skip_wrong_shape: bool = False) -> None:
        """Loads models state dict from the specified path.

        Args:
            loadpath: the path from which the state dict will be loaded.
            skip_wrong_shape: If False, will raise an exception if checkpoints
                weigths shape doesn't match models weights shape.
                If True, will skip unmatched weights and load only matched.
        """
        loadpath = Path(loadpath)
        checkpoint = torch.load(loadpath,
                                map_location=lambda storage, loc: storage)

        # workaround DataParallelModel
        if not isinstance(self.state.core.model, DataParallelModel) \
                and "module." in list(checkpoint.keys())[0]:
            # [7:] is to skip 'module.' in group name
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}

        load_state_dict(model=self.state.core.model,
                        state_dict=checkpoint,
                        skip_wrong_shape=skip_wrong_shape)

    def to_device(self,
                  batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Moves tensors in batch to self.device.

        Args:
            batch: the batch dict.

        Returns:
            The batch dict with tensors on self.device.
        """
        res = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                v = self.to_device(v)
            else:
                if hasattr(v, "to"):
                    v = v.to(self.device)
            res[k] = v

        return res

    def to_fp16(self):
        """Use NVIDIA apex library for mixed precision training.
        After calling this method, all operations will be used in mixed precision.

        Returns:
            self
        """
        self.state.core.model = amp.initialize(self.state.core.model,
                                               opt_level="O1",
                                               verbosity=0)
        self.state.core.use_fp16 = True
        return self

    def set_mode(self, mode: str) -> None:
        """Set the model to train or val and switch dataloaders

        Args:
            mode: 'train', 'val' or 'test', the mode of training procedure.
        """
        if mode == "train":
            self.state.core.model.train()
            self.state.core.loader = self.state.core.dataowner.train_dl
        elif mode == "val":
            self.state.core.model.eval()
            self.state.core.loader = self.state.core.dataowner.val_dl
        elif mode == "test":
            self.state.core.model.eval()
            self.state.core.loader = self.state.core.dataowner.test_dl
        self.state.core.mode = mode

    def freeze_to(self,
                  n: int,
                  freeze_bn: bool = False,
                  model_attr: Optional[str] = None) -> None:
        """Freeze model or model's part till specified layer.

        Args:
            n: the layer number to freeze to
            freeze_bn: if True batchnorm layers will be frozen too
            model_attr: the name of the model attribute if you want to specify
                when you want to freeze layers.
                For examples see example ipynb's.
        """

        module = self.get_model_attr(model_attr)
        freeze_to(module, n, freeze_bn)

    def freeze(self,
               freeze_bn: bool = False,
               model_attr: Optional[str] = None) -> None:
        """Freeze model or model's part till the last layer

        Args:
            freeze_bn: if True batchnorm layers will be frozen too
            model_attr: the name of the model attribute if you want to specify
                when you want to freeze layers.
                For examples see example ipynb's.
        """
        module = self.get_model_attr(model_attr)
        freeze(module, freeze_bn)

    def unfreeze(self,
                 model_attr: Optional[str] = None) -> None:
        """Unfreeze all model or model's part layers.

        Args:
            model_attr: the name of the model attribute if you want to specify
                when you want to freeze layers.
                For examples see example ipynb's.
        """
        module = self.get_model_attr(model_attr)
        unfreeze(module)

    def get_model_attr(self, model_attr: Union[str, None]) -> torch.nn.Module:
        """Get models attribute by name or return the model if name is None.

        Args:
            model_attr: models attribute name to get. If none, than the model
                will be returned.

        Returns:
            The models attribute or the model itself.
        """
        if self.state.core.parallel:
            model = self.state.core.model.module
        else:
            model = self.state.core.model

        if model_attr is not None:
            module = getattr(model, model_attr)
        else:
            module = model
        return module

    def add_callbacks(self, callbacks: List[Callback]) -> None:
        """Add callbacks to the beginning of self.callbacks"""
        self.callbacks = Callbacks(callbacks + self.callbacks.callbacks)

    @staticmethod
    def plot_kek(logdir: Union[str, Path],
                 step: Optional[str] = "batch",
                 metrics: Optional[List[str]] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None) -> None:
        """Plots your keks results in Jupyter Notebook.

        Args:
            logdir: the logdir that was specified during kek.
            step: 'batch' or 'epoch' - what logs to show: for batches or
                for epochs
            metrics: list of metrics to plot. The loss should be specified as
                'loss', learning rate = 'lr' and other metrics should be
                specified as names in metrics dict that was specified during kek
            height: the height of the whole resulting plot
            width: the width of the whole resulting plot

        """
        assert step in ["batch", "epoch"], f"Step should be either 'batch' or" \
                                           f"'epoch', got '{step}'"
        plot_tensorboard_log(logdir, step, metrics, height, width)

    @staticmethod
    def plot_kek_lr(logdir: Union[str, Path],
                    height: Optional[int] = None,
                    width: Optional[int] = None) -> None:
        """Plots learing rate finding results in Jupyter Notebook
        Args:
            logdir: the logdir that was specified during kek_lr.
            height: the height of the whole resulting plot
            width: the width of the whole resulting plot
        """
        step = "batch"
        metrics = ["loss", "lr"]
        plot_tensorboard_log(logdir, step, metrics, height, width)

    @property
    def is_train(self) -> bool:
        return self.state.core.mode == "train"
