import cmath
from pathlib import Path
import io
from typing import Any, Union

from networkx import center
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torchvision as tv

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import torch
import numpy as np

import cv2
from PIL import Image


class Visualize:

    def __init__(
        self,
        slice_vis_range: int = 3,
        figsize: tuple[int, int] = (20, 10),
        hist_norm: bool = True,
        save: bool = False,
        path: Path = "",
    ):
        self.slice_vis_range = slice_vis_range
        self.figsize = figsize
        self.hist_norm = hist_norm
        self.save = save
        self.path = path

    def plot_to_image(self, figure):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(figure)
        plt.axis("off")
        buf.seek(0)
        image = Image.open(buf)
        image = tv.transforms.ToTensor()(image)
        image = image.permute(1, 2, 0)
        return image

    def save_figure(self, figure, name: str):
        figure.savefig(self.path / (name + ".png"))
        return None

    def vis_soft_hot(self, soft_hot: np.array, center_slice: Union[int, None] = 0, linewidth: int = 10):

        # remove batch
        if soft_hot.ndim == 2:
            soft_hot = soft_hot[0]
        soft_hot = soft_hot.squeeze()

        figure = plt.figure(figsize=self.figsize)
        if center_slice is not None:
            plt.axvline(center_slice, color="red", linestyle="--", linewidth=linewidth)
        plt.plot(soft_hot, linewidth=linewidth)

        return self.plot_to_image(figure)

    def vis_image(
        self,
        mammo: Union[torch.Tensor, np.array],
        dbt: Union[torch.Tensor, np.array],
        soft_hot: Union[torch.Tensor, np.array],
        center_slice: Union[int, None] = 0,
        dpi: int = 100,
        fontsize: int = 30,
    ):
        # BCS_DBT does not have mammo, so create a dummy mammo
        if mammo is None:
            mammo = torch.ones(dbt.shape[-2], dbt.shape[-1])

        # assure numpy
        if not isinstance(mammo, np.ndarray):
            mammo = np.array(mammo)  # torch to np

        # no channel dim
        if len(mammo.shape) == 3:
            mammo = mammo[0]  # remove channel

        # assure numpy
        if not isinstance(dbt, np.ndarray):
            dbt = np.array(dbt)  # torch to np

        # no channel dim
        if len(dbt.shape) == 4:
            dbt = dbt[0]

        # make matplotlib display friendly
        mammo = self.process_slice(mammo)
        dbt = compose(np.stack, list, map(self.process_slice))(dbt)

        # select slice indices to visualize
        D = dbt.shape[0]
        max_index = np.argmax(soft_hot)
        slice_start = max(0, max_index - (self.slice_vis_range // 2))
        slice_end = min(D - 1, max_index + (self.slice_vis_range // 2))
        slices = list(range(slice_start, slice_end + 1))
        # handle edge case
        if len(slices) < self.slice_vis_range:
            slices = [0] + slices

        figure, axs = plt.subplots(2, self.slice_vis_range + 1, figsize=self.figsize, dpi=dpi, tight_layout=True)
        axs[0, 0].imshow(dbt[center_slice], interpolation="none", cmap="gray")
        axs[0, 0].axis("off")
        axs[0, 0].set_title(f"center DBT: {center_slice}", fontsize=fontsize)
        axs[1, 0].imshow(mammo, interpolation="none", cmap="gray")
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Mammo", fontsize=fontsize)
        for r in range(2):
            for c in range(self.slice_vis_range):
                if r == 0:
                    try:
                        dbt_slice = dbt[slices[c]]
                    except:
                        dbt_slice = np.ones_like(mammo, dtype=np.uint8) * 255

                    axs[r, c + 1].imshow(dbt_slice, interpolation="none", cmap="gray")
                    axs[r, c + 1].set_title(f"Pred DBT:{slices[c]}", fontsize=fontsize)
                    axs[r, c + 1].axis("off")
                else:
                    slice_n = np.random.choice(len(dbt))
                    dbt_slice = dbt[slice_n]
                    axs[r, c + 1].imshow(dbt_slice, interpolation="none", cmap="gray")
                    axs[r, c + 1].axis("off")
                    axs[r, c + 1].set_title(f"Random DBT: {slice_n}", fontsize=fontsize)
        plt.tight_layout()

        return self.plot_to_image(figure)

    def process_slice(self, slice):

        slice = cv2.normalize(slice, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        slice = slice.astype(np.uint8)
        slice = cv2.equalizeHist(slice)

        return slice


if __name__ == "__main__":

    from glob import glob
    from pathlib import Path
    import random

    mammos = glob("../data/processed/embed/mammo_patch/*.npy")
    mammo = random.sample(mammos, k=1)[0]
    dbt = mammo.replace("mammo_patch", "dbt_patch")

    mammo = np.load(mammo)
    dbt = np.load(dbt).astype(np.float32)

    slice_vis_range = 3
    k_hot = torch.zeros(len(dbt))
    k_hot[16] = 1.0
    center_slice = 32

    vis = Visualize(slice_vis_range=slice_vis_range, save=False, path="")
    img = vis.vis_image(mammo, dbt, k_hot, center_slice=center_slice)
    plt.imshow(img)
    plt.show()

    vis = Visualize(save=False, path="")
    img = vis.vis_soft_hot(k_hot, center_slice=center_slice)
    plt.imshow(img)
    plt.show()
