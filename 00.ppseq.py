# %% import and definition
import os

import numpy as np
import plotly.express as px
import torch
import xarray as xr
from ppseq.model import PPSeq

from routine.io import load_spks
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
    num_templates=1,
    num_neurons=int(spk.sizes["cell"]),
    template_duration=150,
    alpha_a0=1.5,
    beta_a0=0.2,
    alpha_b0=1,
    beta_b0=0.1,
    alpha_t0=1.2,
    beta_t0=0.1,
)
lps, amplitudes = model.fit(spk_dat, num_iter=100)

# %% plotting
fig_temp = ppseq_plot_temp(spk_dat.cpu(), model, amplitudes.cpu())
fig_temp.write_html(os.path.join(FIG_PATH, "temps.html"))
fig_scatter = ppseq_plot_scatter(spk_dat.cpu(), model, amplitudes.cpu())
fig_scatter.write_html(os.path.join(FIG_PATH, "scatter.html"))
