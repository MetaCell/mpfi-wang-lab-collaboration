# %% imports and definition
import os
import shutil

import dask as da
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from minian.cnmf import update_temporal

IN_PATH = "./intermediate/concat"
OUT_PATH = "./intermediate/deconv"
FIG_PATH = "./figs/deconv"
MINIAN_INT = "./minian_int"
PARAM_NCELL_PLT = 10

# %% run deconvolution
if __name__ == "__main__":
    shutil.rmtree(MINIAN_INT, ignore_errors=True)
    os.environ["MINIAN_INTERMEDIATE"] = MINIAN_INT
    os.makedirs(IN_PATH, exist_ok=True)
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(FIG_PATH, exist_ok=True)
    os.makedirs(MINIAN_INT, exist_ok=True)
    sig_master = xr.open_dataarray(os.path.join(IN_PATH, "sig_master.nc")).rename(
        {"master_uid": "unit_id"}
    )
    A_master = xr.open_dataarray(os.path.join(IN_PATH, "A_master.nc")).rename(
        {"master_uid": "unit_id"}
    )
    C_ls = []
    S_ls = []
    for anm, sig_anm in sig_master.groupby("animal"):
        A_anm = A_master.sel(animal=anm).chunk(
            {"height": -1, "width": -1, "unit_id": 1}
        )
        sig_anm = sig_anm.squeeze().chunk({"frame": -1, "unit_id": 1})
        with da.config.set(scheduler="processes"):
            C_anm, S_anm, b0, c0, g, mask = update_temporal(
                A=A_anm,
                C=sig_anm,
                YrA=sig_anm,
                noise_freq=0.1,
                jac_thres=0.8,
                sparse_penal=0.5,
            )
        C_anm = C_anm.assign_coords(session=sig_anm.coords["session"], animal=anm)
        S_anm = S_anm.assign_coords(session=sig_anm.coords["session"], animal=anm)
        np.random.seed(42)
        uid_plt = np.sort(np.random.choice(C_anm.coords["unit_id"], PARAM_NCELL_PLT))
        C_df = C_anm.sel(unit_id=uid_plt).to_series().rename("val").reset_index()
        S_df = S_anm.sel(unit_id=uid_plt).to_series().rename("val").reset_index()
        sig_df = sig_anm.sel(unit_id=uid_plt).to_series().rename("val").reset_index()
        C_df["var"] = "C"
        S_df["var"] = "S"
        sig_df["var"] = "sig"
        plt_df = pd.concat([C_df, S_df, sig_df], ignore_index=True)
        fig = px.line(plt_df, x="frame", y="val", color="var", facet_row="unit_id")
        fig.update_yaxes(matches=None)
        fig.update_layout(height=200 * PARAM_NCELL_PLT)
        fig.write_html(os.path.join(FIG_PATH, "{}.html".format(anm)))
        C_ls.append(C_anm)
        S_ls.append(S_anm)
    C = xr.concat(C_ls, "animal")
    S = xr.concat(S_ls, "animal")
    C.to_netcdf(os.path.join(OUT_PATH, "C.nc"))
    S.to_netcdf(os.path.join(OUT_PATH, "S.nc"))
