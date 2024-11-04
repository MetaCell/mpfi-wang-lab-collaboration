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
