from typing import Optional, Type

import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


# https://github.com/fastai/fastai/blob/e8c855ac70d9d1968413a75c7e9a0f149d28cab3/fastai/layers.py#L171
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self,
                 size: Optional[int] = None):
        "Output will be 2*size or 2 if size is None"
        super().__init__()
        size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x: Type[torch.Tensor]) -> Type[torch.Tensor]:
        return torch.cat([self.mp(x), self.ap(x)], 1)
