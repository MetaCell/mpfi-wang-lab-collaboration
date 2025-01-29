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


def load_bin(
    p_root,
    dat_name="data.bin",
    ops_name="ops.npy",
    downsample=dict(),
    crop=False,
    subset=None,
    verbose=True,
):
    p_ops, p_data = os.path.join(p_root, ops_name), os.path.join(p_root, dat_name)
    ops = np.load(p_ops, allow_pickle=True).item()
    shape = ops["nframes"], ops["Ly"], ops["Lx"]
    data = xr.DataArray(
        np.memmap(p_data, mode="r", dtype="int16", shape=shape),
        dims=["frame", "height", "width"],
        coords={
            "frame": np.arange(shape[0]),
            "height": np.arange(shape[1]),
            "width": np.arange(shape[2]),
        },
    )
    if downsample is not None:
        ds = {
            "frame": slice(0, shape[0], downsample.get("frame", 1)),
            "height": slice(0, shape[1], downsample.get("height", 1)),
            "width": slice(0, shape[2], downsample.get("width", 1)),
        }
        data = data.sel(ds)
    if crop:
        x, y = ops["xrange"], ops["yrange"]
        data = data.sel(height=slice(*y), width=slice(*x))
        if verbose:
            print(f"INFO: Cropped to x-range {x} and y-range {y}")
    bad_frames = ops["badframes"]
    if n := bad_frames.sum():
        if verbose:
            print(f"INFO: found {n} bad frames, but keeping them for now")
    if subset is not None:
        data = data.sel(subset)
    return data


def load_footprint(fpath, height, width):
    stat = np.load(fpath, allow_pickle=True)
    A = np.zeros((len(stat), height, width))
    for i, s in enumerate(stat):
        try:
            A[i, s["ypix"], s["xpix"]] = s["lam"]
        except IndexError:
            print(s["xpix"].max(), s["ypix"].max(), height, width)
    return xr.DataArray(
        A,
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": np.arange(len(stat)),
            "height": np.arange(height),
            "width": np.arange(width),
        },
    )
