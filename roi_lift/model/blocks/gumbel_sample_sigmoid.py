from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
from torch import nn


class GumbelSample(nn.Module):

    def __init__(
        self,
        tau: float = 2.0,
    ):
        super().__init__()

        self.tau = tau

    def forward(
        self,
        logits: "[B, D]",
    ) -> tuple["[B, D]", "[B, D]"]:

        cur_device = logits.device
        self.cur_device = cur_device

        # add gumbel noise
        flatten_logits = logits.flatten(1)
        log_p = flatten_logits

        # simulate sampling
        noisy_logits = log_p + self._get_gumbel_noise(log_p.shape).to(logits.device)
        soft_hots = torch.nn.functional.sigmoid(noisy_logits / self.tau)

        hard_hots = torch.zeros_like(soft_hots, device=cur_device)
        hard_hots[soft_hots > 0.5] = 1

        return hard_hots - soft_hots.detach() + soft_hots, soft_hots

    def get_gumbel_noise(self, shape):

        uniform1 = torch.rand(shape, device=self.cur_device)
        uniform2 = torch.rand(shape, device=self.cur_device)

        return -torch.log(
            torch.log(uniform2 + 1e-20) / torch.log(uniform1 + 1e-20) + 1e-20
        )

    def _get_gumbel_noise(self, shape):
        logistic = torch.rand(shape)
        logistic = logistic.div_(1.0 - logistic).log_()  # ~Logistic(0,1)
        return logistic


if __name__ == "__main__":

    # Example usage
    logits = torch.ones(10000, 1) + 1
    sampler = GumbelSample(tau=2.0)
    hard_hots, soft_hots = sampler(logits)
