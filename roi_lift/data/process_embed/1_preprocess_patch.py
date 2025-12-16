import os
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from pathlib import Path

import pandas as pd
import numpy as np

from ast import literal_eval as eval

import pydicom
import mapply

import cv2

from pydicom.pixel_data_handlers.util import apply_voi_lut

LOAD_DIR: Path = Path(os.path.abspath("[where/you/have/downloaded/embed/data]/"))
SAVE_DIR: Path = Path(os.path.abspath("../processed/embed/"))

# patch base path
MAMMO_PATCH_SAVE_DIR = SAVE_DIR / "mammo_patch"
DBT_PATCH_SAVE_DIR = SAVE_DIR / "dbt_patch"
MAMMO_SAVE_DIR = SAVE_DIR / "mammo_full"


# this is good enough?
# try using bigger image size if the model performance is bad
PATCH_SHAPE = [256, 256]


# Get DICOM image metadata
class DCM_Tags:
    def __init__(self, img_dcm):
        try:
            self.laterality = img_dcm.ImageLaterality
        except AttributeError:
            self.laterality = np.nan

        try:
            self.view = img_dcm.ViewPosition
        except AttributeError:
            self.view = np.nan

        try:
            self.orientation = img_dcm.PatientOrientation
        except AttributeError:
            self.orientation = np.nan


# Check whether DICOM should be flipped
def get_flip_rule(imgdcm):
    # Get DICOM metadata
    tags = DCM_Tags(imgdcm)

    # If image orientation tag is defined
    if ~pd.isnull(tags.orientation):
        # CC view
        if tags.view == "CC":
            if tags.orientation[0] == "P":
                flipHorz = True
            else:
                flipHorz = False

            if (tags.laterality == "L") & (tags.orientation[1] == "L"):
                flipVert = True
            elif (tags.laterality == "R") & (tags.orientation[1] == "R"):
                flipVert = True
            else:
                flipVert = False

        # MLO or ML views
        elif (tags.view == "MLO") | (tags.view == "ML"):
            if tags.orientation[0] == "P":
                flipHorz = True
            else:
                flipHorz = False

            if (tags.laterality == "L") & ((tags.orientation[1] == "H") | (tags.orientation[1] == "HL")):
                flipVert = True
            elif (tags.laterality == "R") & ((tags.orientation[1] == "H") | (tags.orientation[1] == "HR")):
                flipVert = True
            else:
                flipVert = False

        # Unrecognized view
        else:
            flipHorz = False
            flipVert = False

    # If image orientation tag is undefined
    else:
        # Flip RCC, RML, and RMLO images
        if (tags.laterality == "R") & ((tags.view == "CC") | (tags.view == "ML") | (tags.view == "MLO")):
            flipHorz = True
            flipVert = False
        else:
            flipHorz = False
            flipVert = False

    return flipHorz, flipVert


def flip_img(image, flipHorz, flipVert):

    def flip_slice(image):
        if flipHorz:
            image = image[:, ::-1]
        if flipVert:
            image = image[::-1]
        return image

    if len(image.shape) == 2:
        return flip_slice(image)

    else:
        return np.stack([flip_slice(img) for img in image], axis=0)


def flip_roi(roi, image_shape, flipHorz, flipVert):

    if roi and len(roi) == 4:
        # roi: [ymin, xmin, ymax, xmax]
        H, W = image_shape

        ymin, xmin, ymax, xmax = roi

        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(H, ymax)
        xmax = min(W, xmax)

        if flipHorz:
            xmin_new = W - xmax
            xmax_new = W - xmin
            xmin, xmax = min(xmin_new, xmax_new), max(xmin_new, xmax_new)
        if flipVert:
            ymin_new = H - ymax
            ymax_new = H - ymin
            ymin, ymax = min(ymin_new, ymax_new), max(ymin_new, ymax_new)
        return [ymin, xmin, ymax, xmax]
    else:
        return roi


def extract_foreground(img: np.array):

    def np_CountUpContinuingOnes(b_arr: np.array):
        # indice continuing zeros from left side.
        # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
        left = np.arange(len(b_arr))
        left[b_arr > 0] = 0
        left = np.maximum.accumulate(left)

        # from right side.
        # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
        rev_arr = b_arr[::-1]
        right = np.arange(len(rev_arr))
        right[rev_arr > 0] = 0
        right = np.maximum.accumulate(right)
        right = len(rev_arr) - 1 - right[::-1]

        return right - left - 1

    img_copy = img.copy()

    if len(img.shape) == 3:
        img = img[..., 0]

    img = np.where(img <= 40, 0, img)  # To detect backgrounds easily
    height, *_ = img.shape

    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    return img_copy[row_ind][:, col_ind]


def extract_array(dcm):

    def apply_lut(dcm, index=0):
        display_data = apply_voi_lut(dcm.pixel_array, dcm, index=index)
        return display_data

    img = apply_lut(dcm)
    img = img.astype(np.float32)
    img = (65535 * ((img - img.min()) / np.ptp(img))).astype(np.uint16)

    return img


def extract_patch(row) -> tuple[np.array, np.array]:

    patch_id = row["patch_id"]

    if (DBT_PATCH_SAVE_DIR / f"{patch_id}.npy").exists() and (MAMMO_PATCH_SAVE_DIR / f"{patch_id}.npy").exists():
        return row

    else:
        dcm_path_dbt = row["dcm_path_dbt"]
        dcm_path_mammo = row["dcm_path_mammo"]

        # update data path
        # NOTE: this may give eror in newer EMBED dataset
        dcm_path_dbt = (dcm_path_dbt).replace("/mnt/NAS3/mammo/", str(LOAD_DIR))
        dcm_path_mammo = (dcm_path_mammo).replace("/mnt/NAS3/mammo/", str(LOAD_DIR))

        # read dcm
        dbt_dcm = pydicom.dcmread(dcm_path_dbt)
        mammo_dcm = pydicom.dcmread(dcm_path_mammo)

        # dbtdcm does not have laterality and viewposition info
        # so we copy from dcm of mammo
        dbt_dcm.ImageLaterality = mammo_dcm.ImageLaterality
        dbt_dcm.ViewPosition = mammo_dcm.ViewPosition

        mammo_flip_rule = get_flip_rule(mammo_dcm)
        dbt_flip_rule = get_flip_rule(dbt_dcm)

        mammo_array = extract_array(mammo_dcm)
        dbt_array = extract_array(dbt_dcm)

        mammo_array: np.array = flip_img(mammo_array, *mammo_flip_rule)
        dbt_array: np.array = flip_img(dbt_array, *dbt_flip_rule)

        roi = flip_roi(eval(row["rois"])[0], mammo_array.shape, *mammo_flip_rule)

        ymin, xmin, ymax, xmax = roi

        # Resize dbt_array to match mammo_array shape (height, width)
        dbt_array = [
            cv2.resize(
                slice,
                mammo_array.shape[::-1],
                interpolation=cv2.INTER_LINEAR,
            )
            for slice in dbt_array
        ]
        dbt_array = np.array(dbt_array)

        dbt_patch = dbt_array[:, ymin:ymax, xmin:xmax]
        mammo_patch = mammo_array[ymin:ymax, xmin:xmax]

        try:
            dbt_patch = [
                cv2.resize(
                    slice,
                    PATCH_SHAPE,
                    interpolation=cv2.INTER_LINEAR,
                )
                for slice in dbt_patch
            ]
            dbt_patch = np.array(dbt_patch)
            mammo_patch = cv2.resize(
                mammo_patch,
                PATCH_SHAPE,
                interpolation=cv2.INTER_LINEAR,
            )

            np.save(DBT_PATCH_SAVE_DIR / f"{patch_id}.npy", dbt_patch)
            np.save(MAMMO_PATCH_SAVE_DIR / f"{patch_id}.npy", mammo_patch)
        except:
            print("Error extracting patch for patch_id:", patch_id)
            print("DBT array shape:", dbt_array.shape)
            print("Mammo array shape:", mammo_array.shape)
            print("DBT patch shape:", dbt_patch.shape)
            print("Mammo patch shape:", mammo_patch.shape)
            print("ROI coordinates (ymin, xmin, ymax, xmax):", ymin, xmin, ymax, xmax)
            print("Exception occurred.\n")

        return row


def extract_mammo(row):

    patch_id = row["patch_id"]

    if (MAMMO_SAVE_DIR / f"{patch_id}.npy").exists():
        return row

    else:
        dcm_path_mammo = row["dcm_path_mammo"]
        dcm_path_mammo = (dcm_path_mammo).replace("/mnt/NAS3/mammo/", str(LOAD_DIR))

        mammo_dcm = pydicom.dcmread(dcm_path_mammo)
        mammo_flip_rule = get_flip_rule(mammo_dcm)
        mammo_array = extract_array(mammo_dcm)
        mammo_array: np.array = flip_img(mammo_array, *mammo_flip_rule)

        np.save(MAMMO_SAVE_DIR / f"{patch_id}.npy", mammo_array)

        return row


meta_csv = pd.read_csv(SAVE_DIR / "meta.csv")

mapply.init(n_workers=32)
_ = meta_csv.mapply(extract_mammo, axis=1)
_ = meta_csv.mapply(extract_patch, axis=1)
