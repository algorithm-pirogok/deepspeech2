from random import uniform

from torchaudio.transforms import TimeStretch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, p: float, n_freq: int = 128, *args, **kwargs):
        self._p = p
        self._aug = TimeStretch(n_freq=n_freq, *args, **kwargs)  # 3 augmentation

    def __call__(self, data: Tensor):
        if uniform(0, 1) < self._p:
            return self._aug(data)
        return data
