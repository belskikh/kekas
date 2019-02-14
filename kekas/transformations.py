from typing import Callable, Dict, Tuple

import numpy as np
import torch
from torchvision.transforms import Normalize


IMAGENET_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Transformer:
    def __init__(self, key: str, transform_fn: Callable) -> None:
        self.key = key
        self.transform_fn = transform_fn

    def __call__(self,
                 datum: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        datum = datum
        datum[self.key] = self.transform_fn(datum[self.key])
        return datum


def to_torch(scale_factor: float = 255.) -> Callable:
    return lambda x: torch.from_numpy(
        x.astype(np.float32) / scale_factor).permute(2, 0, 1)


def normalize(stats: Tuple = IMAGENET_STATS) -> Normalize:
    return Normalize(*stats)
