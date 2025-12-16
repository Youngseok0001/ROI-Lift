import os
from subprocess import check_output, getoutput
from time import localtime, strftime, time

from toolz import *
from toolz.curried import *

import numpy as np
import torch


def set_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
