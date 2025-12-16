from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
from torch import nn


class SliceSample(nn.Module):

    def __init__(
        self,
        multiply_soft_hot=True,
        soft=False,
    ):
        super().__init__()
        self.multiply_soft_hot = multiply_soft_hot
        self.soft = soft

    def forward(
        self, slices: "[B,C,D,H,W]", hard_hot: "[B,D]", soft_hot: "[B,D]"
    ) -> "[B,C,H,W]":

        if not self.soft:
            merged_slice = torch.einsum(
                "bcdhw,bd -> bchw",
                slices,
                hard_hot * (soft_hot if self.multiply_soft_hot else 1),
            )
        else:
            merged_slice = torch.einsum(
                "bcdhw,bd -> bchw",
                slices,
                soft_hot,
            )

        return merged_slice


if __name__ == "__main__":

    B, D, H, W = 2, 4, 8, 8

    logit = torch.randn(B, D)
    slices = torch.ones(B, 3, D, H, W)

    model = SliceSample()
    merged_slice = model(slices, logit, logit)
