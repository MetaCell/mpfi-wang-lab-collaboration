import numpy as np
import pandas as pd
import ppseq
import ppseq.model
import torch
import torch.nn.functional as F
from ds_utils.utils.num import thres_gmm


def thres_int(a):
    a_th = thres_gmm(a, ncom=3)
    s_amp = np.unique(a_th)[1]
    return np.around(a_th / s_amp)


def classify_neurons(prd):
    cls_cell = np.empty(prd.shape[0])
    for icell, prd_cell in enumerate(prd):
        pval, cts = np.unique(prd_cell[~np.isnan(prd_cell)], return_counts=True)
        if len(cts) > 0:
            cls_cell[icell] = pval[np.argmax(cts)]
        else:
            cls_cell[icell] = -1
    return cls_cell.astype(int)


def sort_neurons(prd, mu, return_df=False):
    """
    Given Neural spike trains X of shape (N,T), return the order (N,) of the neurons
    """
    cls_df = pd.DataFrame(
        {"icell": np.arange(prd.shape[0], dtype=int), "cls": classify_neurons(prd)}
    )
    offsets = np.zeros(len(cls_df))
    for _, row in cls_df.iterrows():
        i, cell_cls = row["icell"], row["cls"]
        if cell_cls >= 0:
            offsets[i] = mu[cell_cls, i]
    cls_df["offset"] = offsets
    cls_df = cls_df.sort_values(["cls", "offset"])
    cls_df["ord"] = np.arange(len(cls_df))
    if return_df:
        return cls_df
    else:
        return np.array(cls_df["icell"])


def predict(
    model: ppseq.model.PPSeq, X: np.ndarray, amp: np.ndarray, return_prob=False
):
    b, W = model.base_rates.cpu(), model.templates.cpu()
    assert amp.shape[0] == W.shape[0], "Number of templates mismatch"
    N, T, D, K = X.shape[0], X.shape[1], W.shape[2], W.shape[0]
    background = b.view(N, 1).expand(N, T)
    prob = np.array(
        [background]
        + [
            F.conv1d(
                amp[[i], :], torch.flip(W[[i]].permute(1, 0, 2), [2]), padding=D - 1
            )[:, : -D + 1]
            for i in range(K)
        ]
    )
    if return_prob:
        return prob
    else:
        prd = prob.argmax(axis=0) - 1
        prd = np.where(X > 0, prd, np.nan)
        return prd
