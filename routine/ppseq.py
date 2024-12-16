import numpy as np
from ds_utils.utils.num import thres_gmm


def thres_int(a):
    a_th = thres_gmm(a, ncom=3)
    s_amp = np.unique(a_th)[1]
    return np.around(a_th / s_amp)
