import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from ds_utils.utils.num import norm
from plotly.colors import qualitative
from scipy.stats import zscore


def sort_neurons(X, scale, mu):
    """
    Given Neural spike trains X of shape (N,T), return the order (N,) of the neurons
    """
    N, T, K = X.shape[0], X.shape[1], scale.shape[0]
    color_list = [[] for _ in range(K)]
    # A, B = [], []
    for i in range(N):
        color_list[int(np.argmax(scale[:, i]))].append(i)
        # if scale[0][i] > scale[1][i]:
        #    A.append(i)
        # else:
        #    B.append(i)
    # A.sort(key=lambda x: mu[0][x])
    # B.sort(key=lambda x: mu[1][x])
    color_list = [
        sorted(color_list[k], key=lambda x: mu[k][x]) for k in range(len(color_list))
    ]
    # return A + B
    return [x for l in color_list for x in l]


def ppseq_color_plot(data, model, amplitudes, pallete=qualitative.Plotly):
    """
    Plot the neural spike trains and
    color the spikes into red, blue and black according to their intensities
    supports at most 30 colors
    """
    b, W, scale, mu = (
        model.base_rates.cpu(),
        model.templates.cpu(),
        model.template_scales.cpu(),
        model.template_offsets.cpu(),
    )
    a = amplitudes
    order = sort_neurons(data, scale, mu)
    N, T, K = data.shape[0], data.shape[1], scale.shape[0]
    D = W.shape[2]
    black_nt = b.view(N, 1).expand(N, T)
    # red_nt = F.conv1d(a[[0], :], torch.flip(
    #    W[[0]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    # blue_nt = F.conv1d(a[[1], :], torch.flip(
    #   W[[1]].permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    # sum_nt = F.conv1d(a, torch.flip(
    # W.permute(1, 0, 2), [2]), padding=D-1)[:, :-D+1]
    # assert torch.allclose(red_nt + blue_nt, sum_nt)
    matrices = np.array(
        [black_nt]
        + [
            F.conv1d(
                a[[i], :], torch.flip(W[[i]].permute(1, 0, 2), [2]), padding=D - 1
            )[:, : -D + 1]
            for i in range(K)
        ]
    )

    def f(i, j):
        if data[i, j] == 0:
            return -1
        else:
            return np.argmax(matrices[:, i, j])

    data_np = data.detach().numpy()
    data_z = np.nan_to_num(np.stack([norm(d) for d in data_np], axis=0))

    # colors = np.array([[f(i, j) for i in range(N)] for j in range(T)])
    colors = np.full((T, N), -1)
    spk_idxs = np.where(data > 0)
    for idx_cell, idx_t in zip(*spk_idxs):
        colors[idx_t, idx_cell] = f(idx_cell, idx_t)
    colors = colors[:, order]

    # black_indices = np.argwhere(colors == 0)
    # red_indices = np.argwhere(colors == 1)
    # blue_indices = np.argwhere(colors == 2)
    color_indices = [np.argwhere(colors == i) for i in range(K + 1)]
    pallete = ["grey"] + pallete

    # Plotting
    fig = go.Figure()
    for i in range(K + 1):
        t, c = color_indices[i][:, 0], color_indices[i][:, 1]
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
            )
        )

    # plt.figure(figsize=(10, 6))
    # for i in range(K + 1):
    #     plt.scatter(
    #         color_indices[i][:, 0],
    #         color_indices[i][:, 1],
    #         c=pallete[i],
    #         s=4,
    #         alpha=0.7,
    #     )
    # plt.scatter(black_indices[:, 0],
    #        black_indices[:, 1], c='black', s=4, alpha=0.7)
    # plt.scatter(red_indices[:, 0], red_indices[:, 1], c='red', s=4, alpha=0.7)
    # plt.scatter(blue_indices[:, 0], blue_indices[:, 1],
    #            c='blue', s=4, alpha=0.7)
    # plt.title("")
    # plt.xlabel("Time")
    # plt.ylabel("channel")
    # plt.grid(True)
    # plt.show()
    return fig
