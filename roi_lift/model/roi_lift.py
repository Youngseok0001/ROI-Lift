import os
import copy
from typing import Sequence

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from roi_lift.model.blocks.slice_sample import SliceSample
from roi_lift.model.blocks.gumbel_sample_sigmoid import GumbelSample
from roi_lift.model.blocks.bread_n_butter import ResidualUnit
from roi_lift.model.backbone.registry import backbone_registry
from roi_lift.model.registry import model_registry
from roi_lift.model.loss.focal_loss import FocalLoss
from roi_lift.util.vis import Visualize


file_dir = os.path.dirname(__file__)


class TVRegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x[:, :-1] - x[:, 1:]).abs().mean()


class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits):

        prob = torch.sigmoid(logits)

        entropy = -(prob * torch.log(prob + self.eps) + (1 - prob) * torch.log(1 - prob + self.eps)).mean()

        return -entropy + 1


@model_registry.register
class RoiLift(nn.Module):

    num_class_d = {"severity": 1, "subtype": 3}

    def __init__(
        self,
        input_shape: Sequence[int] = [128, 128],
        mammo_clip_ckpt_path: str = f"{file_dir}/../../ckpt/b2-model-best-epoch-10.tar",
        tau: float = 0.666,
        min_tau: float = 0.666,
        tv_weight: float = 0.01,
        entropy_weight: float = 0.005,
        severity_weight: float = 1.0,
        subtype_weight: float = 1.0,
        similarity_weight: float = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__()

        ### base attributes ###
        self.input_shape = input_shape

        ### do-not-access ###
        self._tau = tau
        self.min_tau = min_tau
        self._vis = Visualize(save=False, path="", slice_vis_range=3)

        ### derived attributes ###
        self.backbone = backbone_registry["MammoClip"](ckpt_path=mammo_clip_ckpt_path)
        self.frozen_backbone = backbone_registry["MammoClip"](ckpt_path=mammo_clip_ckpt_path)
        self.frozen_backbone.eval()

        self.macro_weight = {}
        self.macro_weight["tv"] = tv_weight
        self.macro_weight["entropy"] = entropy_weight
        self.macro_weight["severity"] = severity_weight
        self.macro_weight["subtype"] = subtype_weight
        self.macro_weight["similarity"] = similarity_weight

        # Freeze the frozen_backbone parameters
        for param in self.frozen_backbone.parameters():
            param.requires_grad = False

        self.input_emb_dim = self.backbone.embedding_dim
        self.output_emb_dim = self.input_emb_dim

        self.embedding_spatial_shape = [i // 32 for i in input_shape]

        self.get_slice_sample = SliceSample()
        self.get_gumbel_sample = GumbelSample(self.tau)

        self.compute_logit = nn.Sequential(
            ResidualUnit(
                spatial_dims=1,
                subunits=2,
                in_channels=self.input_emb_dim,
                out_channels=self.input_emb_dim,
            ),
            ResidualUnit(
                spatial_dims=1,
                subunits=2,
                in_channels=self.output_emb_dim,
                out_channels=self.output_emb_dim // 2,
            ),
            ResidualUnit(
                spatial_dims=1,
                subunits=2,
                in_channels=self.output_emb_dim // 2,
                out_channels=self.output_emb_dim // 2,
            ),
            nn.Conv1d(self.output_emb_dim // 2, 1, 1, 1, 0),
        )

        self.subtype_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                self.output_emb_dim,
                self.output_emb_dim // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_emb_dim // 2, self.num_class_d["subtype"]),
        )

        self.severity_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                self.output_emb_dim,
                self.output_emb_dim // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_emb_dim // 2, self.num_class_d["severity"]),
            nn.Flatten(start_dim=0),
        )

        self.similarity_head = nn.Sequential(
            ResidualUnit(2, self.output_emb_dim, self.output_emb_dim, 1, 1),
            copy.deepcopy(self.frozen_backbone.backbone._bn1),
            copy.deepcopy(self.frozen_backbone.backbone._swish),
        )

        loss_fn_d = {}
        loss_fn_d["subtype"] = FocalLoss(task_type="multi-label")
        loss_fn_d["severity"] = FocalLoss(task_type="binary")
        loss_fn_d["similarity"] = nn.L1Loss()
        loss_fn_d["tv"] = TVRegLoss()
        loss_fn_d["entropy"] = EntropyLoss()
        self.loss_fn_d = nn.ModuleDict(loss_fn_d)

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value
        self.get_gumbel_sample.tau = self._tau

    def forward(self, input_d: dict[str, torch.Tensor]) -> torch.Tensor:

        # the model only takes dbt as input
        dbt: "[B, C, D, H, W]" = input_d["dbt_patch"]
        B, C, D, H, W = dbt.shape

        # flatten depth for faster inference
        dbt_flatten: "[B*D, C, H, W]" = dbt.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # get embedding
        dbt_emb: "[B, C, D, H', W']" = self.backbone(dbt_flatten).reshape(B, D, -1, *self.embedding_spatial_shape).permute(0, 2, 1, 3, 4)

        # compute the unnormalized probability for each slice
        logit: "[B, D]" = self.compute_logit(F.adaptive_avg_pool3d(dbt_emb, (D, 1, 1)).squeeze(-1).squeeze(-1))

        # get slice multi-one-hot
        # hard_hot : "[B, D]" soft_hot : "[B, D]"
        hard_hot, soft_hot = self.get_gumbel_sample(logit)

        # get aggregated slice embedding based on the multi-one-hot
        emb_agg: "[B, C, H', W']" = self.get_slice_sample(dbt_emb, hard_hot, soft_hot)
        # sampl negative aggregated slice embedding for entropy computation
        neg_emb_agg: "[B, C, H', W']" = self.get_slice_sample(dbt_emb, 1 - hard_hot, 1 - soft_hot)

        # map the aggregated embedding to prediction logits
        output_d = {}
        output_d["soft_hot"] = soft_hot
        output_d["subtype_logit"]: "[B, 3]" = self.subtype_head(emb_agg.mean(-1).mean(-1))
        output_d["severity_logit"]: "[B]" = self.severity_head(emb_agg.mean(-1).mean(-1))
        output_d["similarity_logit"]: "[B, C]" = F.adaptive_avg_pool2d(self.similarity_head(emb_agg), 1).squeeze(-1).squeeze(-1)

        output_d["subtype_neg_logit"]: "[B, 3]" = self.subtype_head(neg_emb_agg.mean(-1).mean(-1))
        output_d["severity_neg_logit"]: "[B]" = self.severity_head(neg_emb_agg.mean(-1).mean(-1))

        return merge(output_d, input_d)

    @torch.no_grad()
    def vis(self, input_d):

        # use the first sample in the batch
        if "mammo_patch" in input_d.keys():
            mammo = input_d["mammo_patch"][0].cpu().numpy()
        else:
            # BCS_DBT dataset does not provide mammo
            mammo = None

        dbt = input_d["dbt_patch"][0].cpu().numpy()
        soft_hot = input_d["soft_hot"][0].cpu().numpy()
        center = input_d["slice_center"][0].cpu().item()

        return {
            "vis_vol": self._vis.vis_image(mammo, dbt, soft_hot, center),
            "vis_slice": self._vis.vis_soft_hot(soft_hot, center),
        }

    def compute_loss(
        self,
        output_d,
    ):

        loss_d = {"total_loss": 0}

        # severity loss
        loss_d[f"severity_loss"] = self.loss_fn_d["severity"](output_d[f"severity_logit"], output_d["severity"])
        loss_d["total_loss"] = loss_d[f"severity_loss"] * self.macro_weight["severity"]

        # subtype loss
        loss_d[f"subtype_loss"] = self.loss_fn_d["subtype"](output_d[f"subtype_logit"], output_d["subtype"])
        loss_d["total_loss"] += loss_d[f"subtype_loss"] * self.macro_weight["subtype"]

        # similarity loss
        # similar to perceptual loss, the L1 loss is computed in the embedding space
        full_mammo_emb = self.frozen_backbone(output_d["mammo_full"])
        full_mammo_emb = F.adaptive_avg_pool2d(full_mammo_emb, 1).squeeze(-1).squeeze(-1)
        loss_d[f"similarity_loss"] = self.loss_fn_d["similarity"](full_mammo_emb, output_d["similarity_logit"])
        loss_d["total_loss"] += loss_d[f"similarity_loss"] * self.macro_weight["similarity"]

        # subtype entropy loss
        loss_d[f"entropy_subtype_loss"] = self.loss_fn_d["entropy"](output_d[f"subtype_neg_logit"])
        loss_d["total_loss"] += loss_d[f"entropy_subtype_loss"] * self.macro_weight["entropy"]

        # severity entropy loss
        loss_d[f"entropy_severity_loss"] = self.loss_fn_d["entropy"](output_d[f"severity_neg_logit"])
        loss_d["total_loss"] += loss_d[f"entropy_severity_loss"] * self.macro_weight["entropy"]

        # tv loss
        loss_d[f"tv_loss"] = self.loss_fn_d["tv"](output_d["soft_hot"])
        loss_d["total_loss"] += loss_d[f"tv_loss"] * self.macro_weight["tv"]

        return loss_d

    def set_tau_schedule(self, total_epoch, reduction_mutiplier=0.5):
        # starts from self.tau and attenuates to self.min_tau over total_epoch
        # this function is imported from the original Gumbel-Softmax paper
        self.total_epoch = total_epoch
        self.k = reduction_mutiplier

        self.r = -1 / (max(1, total_epoch * self.k)) * np.log(self.min_tau / self.tau)

    def update_tau_after_epoch(self):
        # update tau after each epoch according to the schedule
        updated_tau = max(self.min_tau, self.tau * np.exp(-self.r))
        self.tau = updated_tau
        return updated_tau


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from pprint import pprint
    from data.embed_loader import EmbedDataset, EmbedLoader
    from itertools import islice

    TEST_FORWARD = False
    TEST_LOSS = False
    TEST_TAU = False
    TEST_SLICE = True

    device = "cuda:6"
    num_slice = 32
    input_shape = [128, 128]

    dataset = EmbedDataset(
        mode="test",
        transform_args={
            "patch_size": input_shape,
            "max_depth": num_slice,
            "full_shape": (1520, 912),
        },
    )

    loader = EmbedLoader(dataset, batch_size=8, num_workers=8, shuffle=False)
    loader = islice(loader, None, 10)
    input_d = next(loader)
    input_d = {k: v.to(device) for k, v in input_d.items()}

    net = RoiLift(
        input_shape=input_shape,
        backbone_name="MammoClip",
        load_ckpt=True,
        tau=2.0,
        min_tau=0.3,
    ).to(device)

    if TEST_FORWARD:
        output_d = net(input_d)
        pprint(output_d.keys())

    if TEST_LOSS:
        output_d = net(input_d)
        loss_d = net.compute_loss(output_d)
        pprint(loss_d.keys())

    if TEST_TAU:
        total_epoch = 100
        net.set_tau_schedule(100)
        taus = []
        for e in range(total_epoch):
            net.update_tau_after_epoch()
            taus.append(net.tau)

        from matplotlib import pyplot as plt

        plt.plot(taus)

    if TEST_SLICE:
        output_d = net(input_d)
        vis_output_d = net.vis(output_d)
        for vis_key, vis_val in vis_output_d.items():
            plt.imshow(vis_val)
            plt.title(vis_key)
            plt.axis("off")
            plt.show()
