import numpy as np
import torch


def np_CountUpContinuingOnes(b_arr):
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


def underestimate_zrange(xs, thres=0.5):

    xs = (np.array(xs) > thres).astype(np.float32)

    max_len = 0
    max_start = -1

    current_len = 0
    current_start = -1

    for i, val in enumerate(xs):
        if val == 1:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
        else:
            current_len = 0

    mask = np.zeros_like(xs)
    if max_start != -1:
        mask[max_start : max_start + max_len] = 1

    return torch.tensor(mask).long()


def overestimate_zrange(xs, thres=0.5):

    xs = (np.array(xs) > thres).astype(np.float32)

    one_locs = np.where(xs == 1)[0]
    first_one_loc = one_locs[0]
    last_one_loc = one_locs[-1]
    mask = np.zeros_like(xs)
    mask[first_one_loc : last_one_loc + 1] = 1

    return torch.tensor(mask).long()


def estimate_zrange(arr, thres=0.5):

    zrange_under = underestimate_zrange(arr, thres=thres).to(torch.float32)
    zrange_over = overestimate_zrange(arr, thres=thres).to(torch.float32)

    one_locs_under = np.where(zrange_under == 1)[0]
    one_locs_over = np.where(zrange_over == 1)[0]
    min_range = int((one_locs_under[0] + one_locs_over[0]) / 2)
    max_range = int((one_locs_under[-1] + one_locs_over[-1]) / 2)

    mask = np.zeros_like(arr)
    mask[min_range : max_range + 1] = 1

    return torch.tensor(mask).long()
