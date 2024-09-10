import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_, PatchEmbed

from .mae_vit import build_2d_sincos_position_embedding

import time

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class ContrastiveDecoder(nn.Module):
    """
    Just a mapping head in order to test the constrastive representations of the encoder
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_contrastive = self.contrastive_pre(x)
        x_contrastive = self.contrastive_head(x_contrastive)
        return x_contrastive
    
class MLP(nn.Module):
    "A MLP for Byol"
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.l1(x).permute(0,2,1)
        x = self.batch_norm(x)
        x = self.relu(x).permute(0,2,1)
        x = self.l2(x)
        return x