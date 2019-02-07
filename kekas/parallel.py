import threading

import torch
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.autograd import Function
import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast


__all__ = ['DataParallelModel', 'DataParallelCriterion']


class Reduce(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, *gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput[0])


class DataParallelModel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelModel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    @staticmethod
    def replicate(module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class DataParallelCriterion(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelCriterion, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, inputs, *targets, **kwargs):
        # input should be already scattered
        # scattering the targets instead
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        inputs = [(input,) for input in inputs]
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, targets, kwargs)
        return self.gather(outputs, self.output_device)

    @staticmethod
    def replicate(module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, targets, kwargs):
        return criterion_parallel_apply(
            replicas, inputs, targets, kwargs, self.device_ids[:len(replicas)])

    @staticmethod
    def gather(outputs, output_device):
        if not len(outputs[0].shape):
            outputs = [output.unsqueeze(0) for output in outputs]
        return Reduce.apply(*outputs) / len(outputs)


def criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(*input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target, kwargs, device),
                                    )
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


def get_a_var(obj):
    if isinstance(obj, torch.tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.tensor):
                return result
    return None