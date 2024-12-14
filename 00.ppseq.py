# %% import and definition
import matplotlib.pyplot as plt
import pandas as pd
import torch
from ppseq.model import PPSeq
from ppseq.plotting import color_plot, plot_model

# %% load data
url = "https://raw.githubusercontent.com/lindermanlab/ppseq-pytorch/main/data/songbird_spikes.txt"
df = pd.read_csv(url, delimiter="\t", header=None, names=["neuron", "time"])
list_of_spiketimes = df.groupby("neuron")["time"].apply(list).to_numpy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bin_width = 0.1
num_timesteps = int(df["time"].max() // bin_width) + 1
num_neurons = len(list_of_spiketimes)
data = torch.zeros(num_neurons, num_timesteps, device=device)
for i, spike_times in enumerate(list_of_spiketimes):
    for t in spike_times:
        data[i, int(t // bin_width)] += 1
data = data[~torch.all(data == 0, dim=1)]

# %% ppseq
torch.manual_seed(0)
model = PPSeq(
    num_templates=2,
    num_neurons=num_neurons,
    template_duration=10,
    alpha_a0=1.5,
    beta_a0=0.2,
    alpha_b0=1,
    beta_b0=0.1,
    alpha_t0=1.2,
    beta_t0=0.1,
)
lps, amplitudes = model.fit(data, num_iter=100)

# %% plotting
plot_model(model.templates.cpu(), amplitudes.cpu(), data.cpu(), spc=0.33)
color_plot(data.cpu(), model, amplitudes.cpu())
