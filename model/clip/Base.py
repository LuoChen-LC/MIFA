import torch
from torch import nn, Tensor


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass


def get_model(model_name):
    if model_name == "clip_b32":
        from .ClipB32 import ClipB32FeatureExtractor
        return ClipB32FeatureExtractor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")