# %% import and definition
import os
import pickle as pkl

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import xarray as xr
from ppseq.model import PPSeq
from scipy.signal import medfilt
from scipy.stats import zscore

from routine.plotting import ppseq_plot_scatter, ppseq_plot_temp
from routine.ppseq import thres_int, unique_temp

F_PATH = "./intermediate/concat/sig_master.nc"
SPK_PATH = "./intermediate/deconv/S.nc"
INT_PATH = "./intermediate/ppseq"
FIG_PATH = "./figs/ppseq"
PARAM_DS = 7
PARAM_UNQ_TOL = 1e-1

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

# %% load data
# spks = xr.open_dataarray(SPK_PATH).squeeze()
# spks_thres = xr.apply_ufunc(
#     thres_int,
#     spks,
#     input_core_dims=[["frame"]],
#     output_core_dims=[["frame"]],
#     vectorize=True,
# ).rename("spks_thres")
# spks_ds = (
#     spks_thres.drop_vars("session")
#     .coarsen({"frame": PARAM_DS}, boundary="trim", coord_func="median")
#     .sum()
#     .rename("spks_ds")
# )
# ds_spks = xr.merge([spks, spks_thres, spks_ds])
# ds_spks.to_netcdf(os.path.join(INT_PATH, "spks_ds.nc"))

# %% ppseq
ds_spks = xr.load_dataset(os.path.join(INT_PATH, "spks_ds.nc")).rename(unit_id="cell")
spk = ds_spks["spks_ds"].dropna("frame", how="all")
spk = spk.where(spk > 2, other=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spk_dat = torch.from_numpy(spk.values)
torch.manual_seed(0)
for ntemp in range(1, 26):
    print("Fitting model with {} templates".format(ntemp))
    model = PPSeq(
        num_templates=ntemp,
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
    with open(os.path.join(INT_PATH, "model-temp{}.pkl".format(ntemp)), "wb") as pklf:
        pkl.dump(
            {"model": model, "X": spk_dat.cpu(), "lps": lps, "amp": amplitudes.cpu()},
            pklf,
        )

# %% plotting
pkl_files = list(filter(lambda fn: fn.endswith(".pkl"), os.listdir(INT_PATH)))
unq_temp_df = []
for pklf in pkl_files:
    with open(os.path.join(INT_PATH, pklf), "rb") as pklf:
        ds = pkl.load(pklf)
    model, amp, X = ds["model"], ds["amp"], ds["X"]
    ntemp = model.num_templates
    model, amp = unique_temp(model, amp, tol=PARAM_UNQ_TOL)
    nunq = model.num_templates
    print("ntemp: {}, nunq: {}".format(ntemp, nunq))
    unq_temp_df.append(pd.DataFrame([{"ntemp": ntemp, "nunq": nunq}]))
    # F = xr.open_dataarray(F_PATH).squeeze()
    # F = (
    #     xr.apply_ufunc(
    #         lambda x: zscore(medfilt(x, kernel_size=31)),
    #         F,
    #         input_core_dims=[["frame"]],
    #         output_core_dims=[["frame"]],
    #         vectorize=True,
    #     )
    #     .drop_vars("session")
    #     .coarsen({"frame": PARAM_DS}, boundary="trim")
    #     .mean()
    #     .clip(0, 5)
    # )
    fig_temp = ppseq_plot_temp(X, model, amp)
    fig_temp.write_html(os.path.join(FIG_PATH, "temps-{}.html".format(ntemp)))
    fig_scatter = ppseq_plot_scatter(X, model, amp)
    fig_scatter.write_html(os.path.join(FIG_PATH, "scatter-{}.html".format(ntemp)))
unq_temp_df = pd.concat(unq_temp_df).sort_values("ntemp").reset_index()
fig = px.line(unq_temp_df, x="ntemp", y="nunq")
fig.write_html(os.path.join(FIG_PATH, "nunq.html"))
