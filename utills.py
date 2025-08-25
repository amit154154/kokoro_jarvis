import torch
import numpy as np
import random

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def str2bool_default_true(s):
    return False if isinstance(s, str) and s.lower() in ("0", "false", "no", "off") else True

def str2bool_default_false(s):
    return False if isinstance(s, str) and s.lower() in ("0", "false", "no", "off") else True