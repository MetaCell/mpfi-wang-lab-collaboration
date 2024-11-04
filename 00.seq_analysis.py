# %% import and definition
import os

import numpy as np
import plotly.express as px
import xarray as xr
from ds_utils.utils.num import norm
from scipy.signal import medfilt
from scipy.stats import zscore
from seqnmf import seqnmf

from routine.io import load_F

IN_DPATH = "./data/ANMP215/A215-20230118/04/suite2p/plane0/"
OUT_PATH = "./intermediate/seqnmf/"
FIG_PATH = "./figs/seqnmf/"
PARAM_FILT_WND = 31
PARAM_NCOMP = 50
PARAM_TLEN = 300
PARAM_DS = 4
PARAM_F_CLIP = (0, 5)

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% seqNMF analysis
F = load_F(IN_DPATH)  # TODO: check why some values are negative
F = (
    xr.apply_ufunc(
        lambda x: zscore(medfilt(x, kernel_size=PARAM_FILT_WND)),
        F,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
    )
    .coarsen({"frame": PARAM_DS}, boundary="trim")
    .mean()
    .clip(*PARAM_F_CLIP)
)
W, H, cost, loadings, power = seqnmf(
    F,
    K=PARAM_NCOMP,
    L=PARAM_TLEN,
    Lambda=1e-4,
    lambda_L1H=0,
    lambda_L1W=0,
    max_iter=10,
    shift=False,
)
W = xr.DataArray(
    W,
    dims=["cell", "comp", "t"],
    coords={
        "cell": F.coords["cell"],
        "comp": np.arange(PARAM_NCOMP),
        "t": np.arange(PARAM_TLEN),
    },
    name="W",
)
H = xr.DataArray(
    H,
    dims=["comp", "frame"],
    coords={"comp": np.arange(PARAM_NCOMP), "frame": F.coords["frame"]},
    name="H",
)
seq_ds = xr.merge([F, W, H])
seq_ds.to_netcdf(os.path.join(OUT_PATH, "seq_ds.nc"))

# %% plotting
seq_ds = xr.open_dataset(os.path.join(OUT_PATH, "seq_ds.nc"))
F, W, H = seq_ds["F"], seq_ds["W"], seq_ds["H"]
H = xr.apply_ufunc(
    norm,
    H,
    input_core_dims=[["frame"]],
    output_core_dims=[["frame"]],
    vectorize=True,
    kwargs={"q": (0, 0.97)},
)
W = xr.apply_ufunc(
    norm,
    W,
    input_core_dims=[["cell", "t"]],
    output_core_dims=[["cell", "t"]],
    vectorize=True,
    kwargs={"q": (0, 0.999)},
)
fig_F = px.imshow(F)
fig_F.write_html(os.path.join(FIG_PATH, "F.html"))
fig_H = px.imshow(H)
fig_H.write_html(os.path.join(FIG_PATH, "H.html"))
fig_W = px.imshow(W, facet_col="comp", facet_col_wrap=5, facet_row_spacing=0.01)
fig_W.update_layout(height=PARAM_NCOMP / 5 * 800)
fig_W.write_html(os.path.join(FIG_PATH, "W.html"))
