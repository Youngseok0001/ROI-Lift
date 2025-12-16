# process the newly introduced 3d roi labels

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import pandas as pd

import mapply
from pydicom import dcmread


def process(config):

    # load dfs
    df_roi = pd.read_csv(config.ORIGINAL_META_PATH)
    df_meta = pd.read_csv(config.PROCESSED_META_PATH)

    ### process meta ###
    df_meta = df_meta[["dcm_path_dbt"]]

    ### process 3d roi df ###
    df_roi = df_roi[
        [
            "anon_dicom_path",
            "ROI_coords",
            "ROI_frames",
        ]
    ]
    df_roi = df_roi.rename(
        columns={
            "anon_dicom_path": "dcm_path_dbt",
            "ROI_coords": "rois",
            "ROI_frames": "roi_frames",
        }
    )

    # only keep rows with non-empty roi_frames
    df_roi = df_roi[df_roi["roi_frames"] != "[]"]

    # merge
    df_roi = pd.merge(df_roi, df_meta, on=["dcm_path_dbt"], how="inner")

    # compute mean of roi_frames
    def compute_mean(xs):
        return int(sum(xs) / len(xs))

    df_roi["roi_frames"] = df_roi["roi_frames"].apply(lambda xss: compute_mean(eval(xss)[0]))

    def get_shape(row):
        dcm_path = row["dcm_path_dbt"]
        img = dcmread(dcm_path)
        arr = img.pixel_array
        return arr.shape[0]

    mapply.init(n_workers=64)
    df_roi["D"] = df_roi.mapply(get_shape, axis=1)

    def get_roi(row):
        D = row["D"]
        center_slie = row["roi_frames"]

        z_min = max(center_slie - (D // 8), 0)
        z_max = min(center_slie + (D // 8), D - 1)

        return [z_min, z_max]

    df_roi["rois"] = df_roi.apply(get_roi, axis=1)

    df_roi = df_roi.drop(columns=["roi_frames"])

    df_roi.to_csv(config.SAVE_NEW_DBT_ROI_PATH, index=False)

    return None


if __name__ == "__main__":

    from easydict import EasyDict as edict
    import os

    # NOTE : set it appropriately
    config = edict()
    config.PROCESSED_META_PATH = os.path.abspath("../processed/embed/meta.csv")
    config.ORIGINAL_META_PATH = os.path.abspath("[where/you/have/downloaded/embed/data]/tables/meta.csv")
    config.SAVE_NEW_DBT_ROI_PATH = os.path.abspath("../embed_new/dbt_roi.csv")

    process(config)
