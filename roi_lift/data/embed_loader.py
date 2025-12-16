import os
from typing import Any, Union
from pathlib import Path
from glob import glob

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import pandas as pd
import numpy as np
import torch

from monai.data import DataLoader
from monai.data.dataset import CacheDataset

from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    RepeatChanneld,
    ToTensord,
    ScaleIntensityd,
    Lambdad,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandSimulateLowResolutiond,
    RandAdjustContrastd,
    RandZoomd,
    Resized,
)

uncurry = lambda f: lambda x: f(*x)


def transform(
    mean=(0.308, 0.308, 0.308),
    std=(0.2505, 0.2505, 0.2505),
    patch_size: tuple[int, int] = (256, 256),
    full_shape: tuple[int, int] = (1520, 912),
    max_depth: int = -1,
    mode: str = ["train", "valid", "test"][0],
    *args,
    **kwargs,
):

    deterministic_transforms = [
        LoadImaged(keys=["mammo_patch", "dbt_patch", "mammo_full"]),
        EnsureChannelFirstd(keys=["mammo_patch", "dbt_patch", "mammo_full"]),
        EnsureTyped(keys=["mammo_patch", "dbt_patch"], data_type="tensor"),
        Resized(keys=["mammo_patch"], spatial_size=[*patch_size]),
        Resized(keys=["dbt_patch"], spatial_size=[max_depth, *patch_size]),
        Resized(keys=["mammo_full"], spatial_size=[*full_shape]),
        Lambdad(keys=["mammo_patch"], func=lambda x: x[None, ...]),
        RepeatChanneld(keys=["mammo_patch", "dbt_patch", "mammo_full"], repeats=3),
        ScaleIntensityd(keys=["mammo_patch", "dbt_patch", "mammo_full"]),
        NormalizeIntensityd(
            keys=["mammo_patch", "dbt_patch", "mammo_full"],
            subtrahend=mean,
            divisor=std,
            channel_wise=True,
        ),
        ToTensord(keys=["mammo_patch", "dbt_patch", "mammo_full"]),
    ]

    random_transforms = [
        RandGaussianNoised(prob=0.1, keys=["mammo_patch", "dbt_patch"]),
        RandGaussianSmoothd(
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            keys=["mammo_patch", "dbt_patch"],
        ),
        RandSimulateLowResolutiond(
            prob=0.25,
            zoom_range=(0.5, 1),
            keys=["mammo_patch", "dbt_patch"],
        ),
        RandAdjustContrastd(
            prob=0.3,
            gamma=(0.7, 1.5),
            invert_image=False,
            retain_stats=True,
            keys=["mammo_patch", "dbt_patch"],
        ),
        RandZoomd(
            keys=["mammo_patch", "dbt_patch"],
            min_zoom=(1, 0.9, 0.9),
            max_zoom=(1, 1.1, 1.1),
            prob=0.5,
        ),
        RandFlipd(keys=["mammo_patch", "dbt_patch"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["mammo_patch", "dbt_patch"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["mammo_patch", "dbt_patch"], factors=(-0.25, 0.25), prob=0.15),
    ]

    return Compose(
        deterministic_transforms
        + (random_transforms if mode == "train" else [])
        + [Lambdad(keys=["mammo_patch"], func=lambda x: x.squeeze(1))]
    )


class EmbedDataset(CacheDataset):

    MAMMO_PATCH_DIRNAME: str = "mammo_patch"
    DBT_PATCH_DIRNAME: str = "dbt_patch"
    MAMMO_FULL_DIRNAME: str = "mammo_full"

    META_PATH: str = "meta.csv"
    ROI_3D_PATH: str = "dbt_roi.csv"

    COLUMNS_OF_INTEREST: list[str] = [
        "patient_id",
        "study_id",
        "patch_id",
        "mass",
        "calcification",
        "arch_distortion",
        "asymmetry",
        "severity",
        "flipped_rois",
        "D",
        "flipped_rois",
        "rois",
        "dcm_path_dbt",
        "dcm_path_mammo",
        "H",
        "W",
        "D",
    ]

    SUBTYPE_COLUMNS: list[str] = [
        "mass",
        "calcification",
        "arch_distortion",
        # "asymmetry",
    ]

    SEVERITY_COLUMN: str = "severity"

    def __init__(
        self,
        base_data_path: Path = Path(os.path.abspath("./processed/embed")),
        mode: str = ["train", "valid", "test"][0],
        transform_args: dict[str, Any] = {
            "mean": (0.308, 0.308, 0.308),
            "std": (0.2505, 0.2505, 0.2505),
            "patch_size": (256, 256),
            "max_depth": -1,
            "full_shape": (1520, 912),
        },
    ):
        # base attrs
        self.base_data_path = Path(base_data_path)
        self.mode = mode
        self.transform_args = transform_args
        self.transform = transform(**transform_args, mode=mode)

        # derived attrs
        datapoints = self.parse()
        datapoints_with_gt = compose(list, filter(lambda pt: pt["roi"][0] != -1))(datapoints)
        datapoints_without_gt = compose(list, filter(lambda pt: pt["roi"][0] == -1))(datapoints)

        # use all datapoint without roi for training
        # split datapoints with roi: half for valid, half for test
        if mode == "train":
            self.datapoints = datapoints_without_gt
        elif mode == "valid":
            self.datapoints = datapoints_with_gt[: len(datapoints_with_gt) // 2]
        elif mode == "test":
            self.datapoints = datapoints_with_gt[len(datapoints_with_gt) // 2 :]
        else:
            raise ValueError

    def parse(self):

        def parse_mammo_patch(row):
            patch_id = row["patch_id"]
            mammo_patch_path = self.base_data_path / self.MAMMO_PATCH_DIRNAME / f"{patch_id}.npy"
            return {"mammo_patch": mammo_patch_path}

        def parse_patch_id(row):
            patch_id = row["patch_id"]
            return {"patch_id": patch_id}

        def parse_dbt_patch(row):
            patch_id = row["patch_id"]
            dbt_full_path = row["dcm_path_dbt"]
            dbt_patch_path = self.base_data_path / self.DBT_PATCH_DIRNAME / f"{patch_id}.npy"
            return {"dbt_patch": dbt_patch_path}

        def parse_mammo_full(row):
            patch_id = row["patch_id"]
            mammo_patch_path = self.base_data_path / self.MAMMO_FULL_DIRNAME / f"{patch_id}.npy"
            return {"mammo_full": mammo_patch_path}

        def parse_subtype(row):
            subtype_d: dict[str, int] = compose(
                valmap(float),
                keyfilter(lambda key: key in self.SUBTYPE_COLUMNS),
            )(row)
            return {"subtype": torch.tensor(list(subtype_d.values()))}

        def parse_severity(row):
            sev_d: dict[str, int] = compose(
                valmap(float),
                keyfilter(lambda key: key == self.SEVERITY_COLUMN),
            )(row)
            return sev_d

        def parse_roi(row):

            H_in, W_in, D_in = row["H"], row["W"], row["D"]
            h_min, w_min, h_max, w_max = eval(row["flipped_rois"])
            full_shape = self.transform_args["full_shape"]

            h_min = max(int(h_min * full_shape[0] / H_in), 0)
            h_max = min(int(h_max * full_shape[0] / H_in), full_shape[0])
            w_min = max(int(w_min * full_shape[1] / W_in), 0)
            w_max = min(int(w_max * full_shape[1] / W_in), full_shape[1])

            if not pd.isna(row["rois"]):
                d_min, d_max = eval(row["rois"])
                max_depth = self.transform_args["max_depth"]
                if max_depth > 0:
                    d_min = int(d_min * max_depth / D_in)
                    d_max = int(d_max * max_depth / D_in)
                return {"roi": torch.tensor([d_min, h_min, w_min, d_max, h_max, w_max])}

            else:
                return {"roi": torch.tensor([-1, h_min, w_min, -1, h_max, w_max])}

        def parse_center(row):
            roi_d = parse_roi(row)
            return {"slice_center": (roi_d["roi"][0] + roi_d["roi"][3]) // 2}

        # merge meta_df and roi_df
        meta_df = pd.read_csv(self.base_data_path / self.META_PATH).drop(columns=["rois"])
        roi_df = pd.read_csv(self.base_data_path / self.ROI_3D_PATH)
        df = pd.merge(meta_df, roi_df, on=["dcm_path_dbt"], how="left")

        # remove cases without image
        valid_patch_ids = [int(Path(s).stem) for s in glob(str(self.base_data_path / self.MAMMO_PATCH_DIRNAME / "*"))]
        df = df[df["patch_id"].isin(valid_patch_ids)]

        # apply parsers to each row
        return compose(
            list,
            map(merge),
            uncurry(zip),
            juxt(
                map(parse_mammo_patch),
                map(parse_dbt_patch),
                map(parse_mammo_full),
                map(parse_subtype),
                map(parse_severity),
                map(parse_patch_id),
                map(parse_roi),
                map(parse_center),
            ),
        )(df.to_dict(orient="records"))

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, i):
        datapoint: dict[str, Any] = self.transform(self.datapoints[i])
        return datapoint


class EmbedLoader(DataLoader):
    def __init__(
        self,
        dataset: CacheDataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        iteration_per_epoch=None,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=None,
        persistent_workers=False,
        *args,
        **kwargs,
    ):

        if iteration_per_epoch is not None:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=iteration_per_epoch)
        else:
            sampler = None

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            *args,
            **kwargs,
        )


if __name__ == "__main__":

    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from roi_lift.util.vis import Visualize

    vis = Visualize(save=False)

    dataset = EmbedDataset(
        mode="test",
        transform_args={
            "patch_size": (256, 256),
            "max_depth": -1,
            "full_shape": (1520, 912),
        },
    )

    for i in tqdm(range(len(dataset))):
        datapoint = dataset[i]
        mammo_patch = datapoint["mammo_patch"]
        dbt_patch = datapoint["dbt_patch"]
        center_slice = datapoint["slice_center"]
        z_min, _, _, z_max, _, _ = datapoint["roi"]
        khot = torch.zeros(dbt_patch.shape[1])
        khot[(z_min + z_max) // 2] = 1.0
        img = vis.vis_image(
            mammo_patch,
            dbt_patch,
            khot,
            center_slice,
        )
        plt.imshow(img)
        plt.show()
