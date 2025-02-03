# %% imports and definition
import os

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from minian.cross_registration import (
    calculate_centroid_distance,
    calculate_centroids,
    calculate_mapping,
    fill_mapping,
    group_by_session,
    resolve_mapping,
)
from minian.motion_correction import apply_transform, estimate_motion
from tqdm.auto import tqdm

from routine.io import load_bin, load_footprint

IN_DPATH = "./data"
IN_SS_CSV = "./data/sessions.csv"
INT_PATH = "./intermediate/concat"
FIG_PATH = "./figs/concat"
PARAM_DIST = 5
PARAM_BASE_Q = 0.05

os.makedirs(INT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


def set_window(wnd):
    return wnd == wnd.min()


def baseline_sub(sig):
    base = np.quantile(sig, PARAM_BASE_Q)
    return sig - base


# %% compute templates and shifts
sscsv = pd.read_csv(IN_SS_CSV)
temps = []
for _, ssrow in tqdm(sscsv.iterrows(), total=len(sscsv)):
    dat = load_bin(os.path.join(IN_DPATH, ssrow["dpath"]))
    temp = (
        dat.max("frame")
        .compute()
        .assign_coords(animal=ssrow["animal"], session=ssrow["session"])
    )
    temps.append(temp.rename("temps"))
temps = xr.combine_nested([temps], ["animal", "session"]).chunk()
shifts = estimate_motion(temps, dim="session").compute().rename("shifts")
temps_sh = apply_transform(temps, shifts).compute().rename("temps_shifted")
window = temps_sh.isnull().sum("session").rename("window")
window, _ = xr.broadcast(window, temps_sh)
window = xr.apply_ufunc(
    set_window,
    window,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    vectorize=True,
)
shift_ds = xr.merge([temps, shifts, temps_sh, window])
fig = px.imshow(shift_ds["temps_shifted"].squeeze(), facet_col="session")
fig.write_html(os.path.join(FIG_PATH, "temps_shifted.html"))
shift_ds.to_netcdf(os.path.join(INT_PATH, "shift_ds.nc"))

# %% apply shifts
sscsv = pd.read_csv(IN_SS_CSV)
shift_ds = xr.open_dataset(os.path.join(INT_PATH, "shift_ds.nc"))
A_shifted = []
for _, ssrow in tqdm(sscsv.iterrows(), total=len(sscsv)):
    anm, ss = ssrow["animal"], ssrow["session"]
    temp = shift_ds["temps"].sel(animal=anm, session=ss)
    # TODO: confirm whether the footprints correspond to cropped movie
    A = load_footprint(
        os.path.join(IN_DPATH, ssrow["dpath"], "stat.npy"),
        temp.sizes["height"],
        temp.sizes["width"],
    )
    sh = shift_ds["shifts"].sel(animal=anm, session=ss)
    A_sh = apply_transform(A, sh)
    A_shifted.append(A_sh)
A_shifted = xr.combine_nested([A_shifted], ["animal", "session"])
A_shifted.to_netcdf(os.path.join(INT_PATH, "A_shifted.nc"))

# %% compute mapping
shift_ds = xr.open_dataset(os.path.join(INT_PATH, "shift_ds.nc"))
A_shifted = xr.open_dataarray(os.path.join(INT_PATH, "A_shifted.nc"))
cents = calculate_centroids(A_shifted, shift_ds["window"])
dist = calculate_centroid_distance(cents, index_dim=["animal"])
dist_ft = dist[dist["variable", "distance"] < PARAM_DIST].copy()
dist_ft = group_by_session(dist_ft)
mappings = calculate_mapping(dist_ft)
mappings_meta = resolve_mapping(mappings)
mappings_meta_fill = fill_mapping(mappings_meta, cents)
mappings_meta_fill.to_pickle(os.path.join(INT_PATH, "mappings_meta_fill.pkl"))

# %% compute master spatial footprint
A_shifted = xr.open_dataarray(os.path.join(INT_PATH, "A_shifted.nc"))
mappings = pd.read_pickle(os.path.join(INT_PATH, "mappings_meta_fill.pkl"))
A_master = []
for anm, map_anm in mappings.groupby(("meta", "animal")):
    A_anm = []
    for uid, Arow in tqdm(map_anm.iterrows(), total=len(map_anm)):
        A_ls = []
        for ss, sid in Arow["session"].dropna().items():
            A_ls.append(
                A_shifted.sel(animal=anm, session=ss, unit_id=sid).drop_vars("unit_id")
            )
        curA = xr.concat(A_ls, "session").sum("session")
        curA = (curA / curA.sum()).assign_coords({"master_uid": uid})
        A_anm.append(curA)
    A_master.append(A_anm)
A_master = xr.combine_nested(A_master, ["animal", "master_uid"])
A_master.to_netcdf(os.path.join(INT_PATH, "A_master.nc"))

# %% extract signals
sscsv = pd.read_csv(IN_SS_CSV)
A_master = xr.open_dataarray(os.path.join(INT_PATH, "A_master.nc"))
shift_ds = xr.open_dataset(os.path.join(INT_PATH, "shift_ds.nc"))
sig_master = []
for anm, Am in A_master.groupby("animal"):
    ss_sub = sscsv[sscsv["animal"] == anm]
    sigs = []
    for _, ssrow in tqdm(sscsv.iterrows(), total=len(sscsv)):
        ss = ssrow["session"]
        dat = load_bin(os.path.join(IN_DPATH, ssrow["dpath"]))
        sh = shift_ds["shifts"].sel(animal=anm, session=ss)
        curA = apply_transform(Am, -sh)
        cur_sig = curA.dot(dat).compute()
        cur_sig = xr.apply_ufunc(
            baseline_sub,
            cur_sig,
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
        )
        sigs.append(cur_sig)
    sigs = xr.concat(sigs, "frame")
    sigs = sigs.assign_coords(frame=np.arange(sigs.sizes["frame"]))
    sig_master.append(sigs)
sig_master = xr.concat(sig_master, "animal").compute()
sig_master.to_netcdf(os.path.join(INT_PATH, "sig_master.nc"))
