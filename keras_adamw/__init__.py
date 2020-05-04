"""Handles version imports
   NOTE: if using `tensorflow.keras` imports, use `os.environ["TF_KERAS"] = '1'`,
         else the default '0' will be assumed for `keras` imports.
"""
import os
import tensorflow as tf

TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
TF_2 = bool(tf.__version__[0] == '2')

if TF_KERAS:
    if TF_2:
        from .optimizers_v2 import AdamW, NadamW, SGDW
    else:
        from .optimizers_225tf import AdamW, NadamW, SGDW
else:
    if TF_2:
        from .optimizers import AdamW, NadamW, SGDW
    else:
        from .optimizers_225 import AdamW, NadamW, SGDW

from .utils_common import get_weight_decays, fill_dict_in_order
from .utils_common import reset_seeds, K_eval


__version__ = '1.2'
