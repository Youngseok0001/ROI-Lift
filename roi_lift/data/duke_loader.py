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
    EnsureTyped,
    EnsureChannelFirstd,
    RepeatChanneld,
    NormalizeIntensityd,
    ToTensord,
    ScaleIntensityd,
    Resized,
)

uncurry = lambda f: lambda x: f(*x)


def transform(
    mean=(0.308, 0.308, 0.308),
    std=(0.2505, 0.2505, 0.2505),
    patch_size: tuple[int, int] = (128, 128),
    max_depth: int = -1,
    *args,
    **kwargs,
):

    # only use deterministic as the datast is only for evaluation purpose
    deterministic_transforms = [
        LoadImaged(keys=["dbt_patch"]),
        EnsureChannelFirstd(keys=["dbt_patch"]),
        EnsureTyped(keys=["dbt_patch"], data_type="tensor"),
        Resized(keys=["dbt_patch"], spatial_size=[max_depth, *patch_size]),
        RepeatChanneld(keys=["dbt_patch"], repeats=3),
        ScaleIntensityd(keys=["dbt_patch"]),
        NormalizeIntensityd(keys=["dbt_patch"], subtrahend=mean, divisor=std, channel_wise=True),
        ToTensord(keys=["dbt_patch"]),
    ]

    return Compose(deterministic_transforms)


class DukeDataset(CacheDataset):

    DBT_PATCH_DIRNAME: str = "dbt_patch"

    META_PATH: str = "meta.csv"

    COLUMNS_OF_INTEREST: list[str] = [
        "patient_id",
        "study_id",
        "laterality",
        "view_position",
        "dcm_path_dbt",
        "rois",
        "patch_id",
    ]
    SUBTYPE_COLUMNS: list[str] = [
        "mass",
        "arch_distortion",
        # "asymmetry", # does not exist in BCS_DBT dataset
        # "calcification", # does not exist in BCS_DBT dataset
    ]

    SEVERITY_COLUMN: str = "severity"

    def __init__(
        self,
        base_data_path: Path = Path(os.path.abspath("./processed/duke")),
        transform_args: dict[str, Any] = {"patch_size": (128, 128), "max_depth": -1},
    ):
        # base attrs
        self.base_data_path = Path(base_data_path)
        self.transform_args = transform_args
        self.transform = transform(**transform_args)

        # derived attrs
        self.datapoints = self.parse()

    def parse(self):

        def parse_roi(row):

            D_in = row["D"]
            z_min, _, _, z_max, _, _ = eval(row["rois"])

            # Scale coordinates to destination size
            if self.transform_args["max_depth"] > 0:
                z_min_out = int(z_min * self.transform_args["max_depth"] / D_in)
                z_max_out = int(z_max * self.transform_args["max_depth"] / D_in)
            else:
                z_min_out = z_min
                z_max_out = z_max

            # fill -1 for x,y coords as we do not use them anyway
            return {"roi": torch.tensor([z_min_out, -1, -1, z_max_out, -1, -1])}

        def parse_center(row):
            roi_d = parse_roi(row)
            return {"slice_center": (roi_d["roi"][0] + roi_d["roi"][3]) // 2}

        def parse_dbt_patch(row):
            patch_id = row["patch_id"]
            dbt_patch_path = self.base_data_path / self.DBT_PATCH_DIRNAME / f"{patch_id}.npy"
            return {"dbt_patch": dbt_patch_path}

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

        df = pd.read_csv(self.base_data_path / self.META_PATH)

        # in case there are no corresponding npy files
        patch_paths = glob(str(self.base_data_path / self.DBT_PATCH_DIRNAME / "*.npy"))
        valid_patch_ids = [int(Path(path).stem) for path in patch_paths]
        df = df[df["patch_id"].isin(valid_patch_ids)]

        # juxtapose all parsing functions
        return compose(
            list,
            map(merge),
            uncurry(zip),
            juxt(
                map(parse_dbt_patch),
                map(parse_subtype),
                map(parse_severity),
                map(parse_roi),
                map(parse_center),
            ),
        )(df.to_dict(orient="records"))

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, data_index):
        datapoint: dict[str, Any] = self.transform(self.datapoints[data_index])
        return datapoint


class DukeLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        iteration_per_epoch: Union[int, None] = None,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=None,
        persistent_workers=False,
        **kwargs,
    ):

        # Initialize the parent class with all arguments
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
            **kwargs,
        )


if __name__ == "__main__":

    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from roi_lift.util.vis import Visualize

    vis = Visualize(save=False)

    dataset = DukeDataset()

    for i in tqdm(range(len(dataset))):
        datapoint = dataset[i]
        dbt_patch = datapoint["dbt_patch"]
        center_slice = datapoint["slice_center"]
        z_min, _, _, z_max, _, _ = datapoint["roi"]
        khot = torch.zeros(dbt_patch.shape[1])
        khot[(z_min + z_max) // 2] = 1.0
        img = vis.vis_image(
            None,
            dbt_patch,
            khot,
            center_slice,
        )
        plt.imshow(img)
        plt.show()
