import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import ppseq
import ppseq.model
import torch
import torch.nn.functional as F
import xarray as xr
from ds_utils.utils.num import norm
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.stats import zscore

from .ppseq import predict, sort_neurons


def ppseq_plot_scatter(
    X: np.ndarray,
    model: ppseq.model.PPSeq,
    amp: np.ndarray,
    pallete=qualitative.Plotly,
    raw_dat: np.ndarray = None,
):
    """
    Plot the neural spike trains and
    color the spikes into red, blue and black according to their intensities
    supports at most 30 colors
    """
    mu = model.template_offsets.cpu()
    K = model.num_templates
    prd = predict(model, X, amp)
    order = sort_neurons(prd, mu)
    data_np = X.detach().numpy()
    data_z = np.nan_to_num(np.stack([norm(d) for d in data_np], axis=0))
    prd = prd[order, :]
    if raw_dat is not None:
        raw_dat = raw_dat[order, :]
        plt_raw = True
    else:
        plt_raw = False
    color_indices = [np.argwhere(prd == i - 1) for i in range(K + 1)]
    pallete = ["grey"] + pallete
    # plotting
    fig = make_subplots(
        rows=3 if plt_raw else 2,
        shared_xaxes=True,
        vertical_spacing=2e-2,
        row_heights=[1, 2, 2] if plt_raw else [1, 2],
    )
    # amplitude plot
    for ia, a in enumerate(np.array(amp)):
        fig.add_trace(
            go.Scatter(y=a.clip(0, 50), mode="lines", name="comp{}".format(ia)),
            row=1,
            col=1,
        )
    # scatter plot
    for i in range(K + 1):
        c, t = color_indices[i][:, 0], color_indices[i][:, 1]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=c,
                marker={
                    "color": pallete[i],
                    # "size": data_z[c, t] * 4,
                    "size": 4,
                    "line": {
                        "width": (data_z[c, t] * 0.2).clip(2),
                        "color": pallete[i],
                    },
                    "symbol": "line-ns",
                },
                mode="markers",
                opacity=1,
                name="comp{}".format(i - 1) if i > 0 else "background",
            ),
            row=2,
            col=1,
        )
    # raw data plot
    if plt_raw:
        fig.add_trace(
            go.Heatmap(
                x=np.array(raw_dat.coords["frame"]),
                z=np.array(raw_dat),
                showscale=False,
            ),
            row=3,
            col=1,
        )
    return fig


def ppseq_plot_temp(X, model, amp):
    prd = predict(model, X, amp)
    cell_ord = sort_neurons(prd, model.template_offsets)
    temp = model.templates.cpu().detach().numpy()
    temp = xr.DataArray(
        temp,
        dims=["temp", "cell", "frame"],
        coords={
            "temp": np.arange(temp.shape[0]),
            "frame": np.arange(temp.shape[2]),
        },
    )
    temp = temp[:, cell_ord, :].assign_coords(cell=np.arange(temp.sizes["cell"]))
    return px.imshow(temp, facet_col="temp")
