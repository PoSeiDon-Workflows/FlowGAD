# %%
import argparse
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU, Softmax
from torch.optim import Adam

from psd_gnn.dataset import PSD_Dataset
from psd_gnn.utils import parse_adj

# %%


class MLP(Module):
    def __init__(self, in_channel, hid_channel, out_channel) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.hid_channel = hid_channel
        self.out_channel = out_channel

        self.fc1 = Linear(self.in_channel, self.hid_channel)
        self.relu1 = ReLU()
        self.fc2 = Linear(self.hid_channel, self.hid_channel)
        self.relu2 = ReLU()
        self.fc3 = Linear(self.hid_channel, self.out_channel)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = F.softmax(x, 1)
        return x


# %%
ts_features = ["ready", "submit", "execute_start", "execute_end", "post_script_start", "post_script_end"]
delay_features = ["wms_delay", "queue_delay", "runtime", "post_script_delay", "stage_in_delay", "stage_out_delay"]

files = glob("./data_new/*/1000*.csv")
single_job = []
label_44 = []
fn = []
for f in files:
    df = pd.read_csv(f, index_col=0)
    single_job.append(df.loc[['individuals_ID0000044']])
    fn.append(f.split("/")[-1:])
#  = fn
single_job = pd.concat(single_job)
# record the file name
single_job['fn'] = fn
# single_job
# subtract the timestamp
single_job[ts_features] = single_job[ts_features].sub(single_job[ts_features].ready, axis="rows")
# NOTE: some files are labeled as None, some are normal
single_job['anomaly_type'].replace({"None": "normal"}, inplace=True)
# fill nan with 0
single_job.fillna(0, inplace=True)
# process hostname

# re.sub("(.*-)?worker(.*-)?(\d)+", "worker-\\3", "poseidon-worker-1")
single_job[['worker', 'container']] = single_job.kickstart_hostname.str.split("-container-", expand=True)
# binary labels of df_y
df_y = single_job['anomaly_type'].map(lambda x: 0 if x in ["normal", "None"] else 1)
# df_y

features = delay_features + ['stage_in_bytes', 'stage_out_bytes', 'kickstart_executables_cpu_time']
Xs, ys = single_job[features].to_numpy(), df_y.to_numpy()

# %%
mlp = MLP(9, 128, 2)
optimizer = Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
loss_func = CrossEntropyLoss()
# normalization
X_ = StandardScaler().fit_transform(Xs)

X_ = torch.tensor(X_, dtype=torch.float32)
y_ = torch.tensor(ys, dtype=torch.long)

# prob_i per epoch
X_probs = []
n_samples = X_.shape[0]

for e in range(200):
    optimizer.zero_grad()
    X_output = mlp(X_)
    X_prob = F.softmax(X_output, dim=1)
    # loss = loss_func(X_prob, y_)
    loss = F.cross_entropy(X_prob, y_)
    loss.backward()
    optimizer.step()
    train_acc = ((X_prob.argmax(1) == y_).sum() / X_prob.shape[0]).item()
    # print(train_acc)
    X_probs.append(X_prob.detach().cpu().numpy()[np.arange(n_samples), ys].reshape(-1, 1))

print(train_acc)
res = np.hstack(X_probs)
fig = plt.figure(figsize=(6, 4), )
gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])
ax0 = fig.add_subplot(gs[0, :])

conf = res[:, 100:].mean(1)
vari = res[:, 100:].std(1)
corr = (res[:, 100:] >= 0.5).sum(1)
# plt.scatter(vari, conf)
pal = sns.diverging_palette(260, 15, sep=10, center="dark")
plot = sns.scatterplot(x=vari, y=conf, ax=ax0, palette=pal, hue=corr / 100, style=corr / 100)


def bb(c):
    return dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")


an1 = ax0.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", rotation=360, bbox=bb('black'))
an2 = ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", bbox=bb('r'))
an3 = ax0.annotate("hard-to-learn", xy=(0.27, 0.25), xycoords="axes fraction", fontsize=12, color='black',
                   va="center", ha="center", bbox=bb('b'))
plot.legend(fancybox=True, shadow=True, ncol=1)

plot.set_xlabel('variability')
plot.set_ylabel('confidence')

plot.set_title("Data Map", fontsize=12)

# Make the histograms.
# ax1 = fig.add_subplot(gs[1, 0])
# ax2 = fig.add_subplot(gs[1, 1])
# ax3 = fig.add_subplot(gs[1, 2])

# # plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
# plott0 = sns.histplot(x=conf, ax=ax1, color='#622a87')
# plott0.set_title('')
# plott0.set_xlabel('confidence')
# plott0.set_ylabel('density')

# # plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
# plott1 = sns.histplot(x=vari, ax=ax2, color='teal')
# plott1.set_title('')
# plott1.set_xlabel('variability')
# plott1.set_ylabel("density")

# plot2 = sns.countplot(x=corr, ax=ax3, color='#86bf91')
# ax3.xaxis.grid(False) # Show the vertical gridlines

# plot2.set_title('')
# plot2.set_xlabel('correctness')
# plot2.set_ylabel('count')

fig.tight_layout()
