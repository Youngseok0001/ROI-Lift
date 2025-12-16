from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics import Metric

from roi_lift.util.misc import (
    overestimate_zrange,
    underestimate_zrange,
    estimate_zrange,
)


class SliceDist(Metric):

    def __init__(
        self,
        dist_mode: str = ["argmax", "standard"][-1],  # use "argmax" for baselines without specific threshold
        *args,
        **kwargs,
    ):

        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.dist_mode = dist_mode

    def update(self, pred, target_center):

        assert len(pred.shape) == 2, "pred must be a 2D tensor with shape (B, D)"
        assert pred.shape[0] == len(target_center), "pred and target must have the same batch size"

        target_center = target_center.to(torch.float32)

        mid_loc_list = [self.get_mid_loc(p) for p in pred]  # list of scalar tensors

        mid_loc = torch.stack(mid_loc_list, dim=0)  # shape (B,)
        abs_diff = (mid_loc - target_center).abs()  # shape (B,)

        self.scores.append(abs_diff)  # always append 1D tensors of shape (B,)

    def compute(self) -> float:
        scores = torch.cat(self.scores)
        return scores.nanmean()

    def get_mid_loc(self, pred):

        if self.dist_mode == "standard":
            pred = (pred > 0.5).to(torch.float32)
            # very rarely there is no positive prediction
            # then, take the middle slice
            if pred.sum() == 0:
                one_locs = torch.where(pred == 0)[0]
                mid_loc = one_locs[len(one_locs) // 2]
            else:
                pred = self.process_pred(pred, thres=0.5)
                one_locs = torch.where(pred == 1)[0]
                mid_loc = one_locs[len(one_locs) // 2]
        else:
            mid_loc = pred.argmax()

        return mid_loc


class SliceDistUnder(SliceDist):
    @staticmethod
    def process_pred(pred, thres=0.5):
        return underestimate_zrange(pred, thres=thres)


class SliceDistOver(SliceDist):
    @staticmethod
    def process_pred(pred, thres=0.5):
        return overestimate_zrange(pred, thres=thres)


class SliceDistAvg(SliceDist):
    @staticmethod
    def process_pred(pred, thres=0.5):
        return estimate_zrange(pred, thres=thres)


if __name__ == "__main__":

    pred = torch.tensor(
        [
            [0, 1, 1, 0, 1, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
        ]
    )
    target = torch.tensor([0, 5])

    slice_dist_under = SliceDistUnder()
    slice_dist_over = SliceDistOver()
    slice_dist_avg = SliceDistAvg()

    slice_dist_under.update(pred, target)
    print(slice_dist_under.compute())

    slice_dist_over.update(pred, target)
    print(slice_dist_over.compute())

    slice_dist_avg.update(pred, target)
    print(slice_dist_avg.compute())
