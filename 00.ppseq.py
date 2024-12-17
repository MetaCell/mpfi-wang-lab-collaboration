# %% import and definition
import os
import pickle as pkl

import numpy as np
import plotly.express as px
import torch
import xarray as xr
from ppseq.model import PPSeq
from scipy.signal import medfilt
from scipy.stats import zscore

from routine.io import load_F, load_spks
from routine.plotting import ppseq_plot_scatter, ppseq_plot_temp
from routine.ppseq import thres_int

IN_DPATH = "./data/ANMP215/A215-20230118/04/suite2p/plane0/"
INT_PATH = "./intermediate/ppseq"
FIG_PATH = "./figs/ppseq"
PARAM_DS = 5

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% load data
spks = load_spks(IN_DPATH)
spks_thres = xr.apply_ufunc(
    thres_int,
    spks,
    input_core_dims=[["frame"]],
    output_core_dims=[["frame"]],
    vectorize=True,
).rename("spks_thres")
spks_ds = (
    spks_thres.coarsen({"frame": PARAM_DS}, boundary="trim", coord_func="median")
    .sum()
    .rename("spks_ds")
)
ds_spks = xr.merge([spks, spks_thres, spks_ds])
ds_spks.to_netcdf(os.path.join(INT_PATH, "spks_ds.nc"))

# %% ppseq
ds_spks = xr.load_dataset(os.path.join(INT_PATH, "spks_ds.nc"))
spk = ds_spks["spks_thres"].dropna("frame", how="all")
spk = spk.where(spk > 2, other=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spk_dat = torch.from_numpy(spk.values)
torch.manual_seed(0)
model = PPSeq(
    num_templates=2,
    num_neurons=int(spk.sizes["cell"]),
    template_duration=100,
    alpha_a0=1.5,
    beta_a0=0.2,
    alpha_b0=1,
    beta_b0=0.1,
    alpha_t0=1.2,
    beta_t0=0.1,
)
lps, amplitudes = model.fit(spk_dat, num_iter=100)
with open(os.path.join(INT_PATH, "model.pkl"), "wb") as pklf:
    pkl.dump(
        {"model": model, "X": spk_dat.cpu(), "lps": lps, "amp": amplitudes.cpu()}, pklf
    )

# %% plotting
with open(os.path.join(INT_PATH, "model.pkl"), "rb") as pklf:
    ds = pkl.load(pklf)
model, amp, X = ds["model"], ds["amp"], ds["X"]
F = load_F(IN_DPATH)
F = (
    xr.apply_ufunc(
        lambda x: zscore(medfilt(x, kernel_size=31)),
        F,
        input_core_dims=[["frame"]],
        output_core_dims=[["frame"]],
        vectorize=True,
    )
    .coarsen({"frame": PARAM_DS}, boundary="trim")
    .mean()
    .clip(0, 5)
)
fig_temp = ppseq_plot_temp(X, model, amp)
fig_temp.write_html(os.path.join(FIG_PATH, "temps.html"))
fig_scatter = ppseq_plot_scatter(X, model, amp, raw_dat=F)
fig_scatter.write_html(os.path.join(FIG_PATH, "scatter.html"))
