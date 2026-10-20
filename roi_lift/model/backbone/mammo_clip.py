import os
from typing import Optional

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
from torch import nn
from roi_lift.third_party.Mammo_CLIP.src.codebase.breastclip.model.modules.efficientnet_custom import (
    EfficientNet,
)

from roi_lift.model.backbone.registry import backbone_registry

file_dir = os.path.dirname(__file__)


@backbone_registry.register
class MammoClip(nn.Module):

    def __init__(
        self,
        ckpt_path: Optional[str] = f"{file_dir}/../../ckpt/b2-model-best-epoch-10.tar",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embedding_dim = 1408

        # ckpt_path=None skips Mammo-CLIP init (e.g., inference, where the ROI-Lift ckpt overwrites every weight)
        if ckpt_path is None:
            self.backbone = EfficientNet.from_name("efficientnet-b2", num_classes=1)
        else:
            self.backbone = EfficientNet.from_pretrained(model_name="efficientnet-b2", num_classes=1)
            self.load_weight(os.path.abspath(ckpt_path))

    def forward(self, x):
        x = self.backbone.extract_features(x)
        return x

    def load_weight(self, mammo_ckpt):
        print(f"Loading mammo weight from {mammo_ckpt}...", end=" ")
        ckpt = torch.load(mammo_ckpt, weights_only=False, map_location="cpu")
        ckpt = compose(
            keymap(lambda x: x.replace("image_encoder.", "")),
            keyfilter(lambda k: "image_encoder" in k),
        )(ckpt["model"])
        self.backbone.load_state_dict(ckpt, strict=False)
        print(f"Loaded")
        return None


if __name__ == "__main__":

    x = torch.randn(1, 3, 256, 256).to("cuda")
    net = MammoClip().to("cuda")
    output = net(x)
