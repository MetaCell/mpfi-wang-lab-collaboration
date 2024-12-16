import os

import numpy as np
import xarray as xr


def load_F(dpath, ret_dff=True):
    f = np.load(os.path.join(dpath, "F.npy"))
    f_neu = np.load(os.path.join(dpath, "Fneu.npy"))
    if ret_dff:
        assert f_neu.min() > 0
        f = f / f_neu
    return xr.DataArray(
        f,
        dims=["cell", "frame"],
        coords={"cell": np.arange(f.shape[0]), "frame": np.arange(f.shape[1])},
        name="F",
    )


def load_spks(dpath, ds=1):
    spks = np.load(os.path.join(dpath, "spks.npy"))
    spks = xr.DataArray(
        spks,
        dims=["cell", "frame"],
        coords={"cell": np.arange(spks.shape[0]), "frame": np.arange(spks.shape[1])},
        name="spks",
    )
    if ds > 1:
        spks = spks.coarsen({"frame": ds}, boundary="trim").sum()
    return spks
