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
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU
from torch.optim import Adam

from psd_gnn.utils import parse_adj

# %%
ts_features = ["ready", "submit", "execute_start", "execute_end", "post_script_start", "post_script_end"]
delay_features = ["wms_delay", "queue_delay", "runtime", "post_script_delay", "stage_in_delay", "stage_out_delay"]

np.random.seed(41)
torch.manual_seed(41)

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
nodes, edges = parse_adj("1000genome_new_2022")
comp_jobs = [node for node in nodes if node.startswith("individuals")]
files = glob("../data_new/*/1000*.csv")

old_job_acc = []
new_job_acc = []
ill_labels = []
for job in sorted(comp_jobs):
    print(job)
    single_job = []
    label_44 = []
    fn = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        single_job.append(df.loc[[job]])
        fn.append(f.split("/")[-1:])
    #
    single_job = pd.concat(single_job)
    # record the file name
    single_job['fn'] = fn

    # subtract the timestamp
    single_job[ts_features] = single_job[ts_features].sub(single_job[ts_features].ready, axis="rows")
    # NOTE: some files are labeled as None, some are normal
    single_job['anomaly_type'].replace({"None": "normal"}, inplace=True)
    # fill nan with 0
    single_job.fillna(0, inplace=True)
    # process hostname

    # REVIEW: re.sub("(.*-)?worker(.*-)?(\d)+", "worker-\\3", "poseidon-worker-1")
    # single_job[['worker', 'container']] = single_job.kickstart_hostname.str.split("-container-", expand=True)

    # binary labels of df_y
    df_y = single_job['anomaly_type'].map(lambda x: 0 if x in ["normal", "None"] else 1)
    # df_y

    features = delay_features + ['stage_in_bytes', 'stage_out_bytes', 'kickstart_executables_cpu_time']
    Xs, ys = single_job[features].to_numpy(), df_y.to_numpy()
    single_job[features]

    mlp = MLP(9, 128, 2)
    optimizer = Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_func = CrossEntropyLoss()
    ''' train with pytorch '''
    # # normalization
    # X_ = StandardScaler().fit_transform(Xs)
    # X_ = torch.tensor(X_, dtype=torch.float32)
    # y_ = torch.tensor(ys, dtype=torch.long)

    # # prob_i per epoch
    # X_probs = []
    # n_samples = X_.shape[0]

    # for e in range(200):
    #     optimizer.zero_grad()
    #     X_output = mlp(X_)
    #     X_prob = F.softmax(X_output, dim=1)
    #     # loss = loss_func(X_prob, y_)
    #     loss = F.cross_entropy(X_prob, y_)
    #     loss.backward()
    #     optimizer.step()
    #     train_acc = ((X_prob.argmax(1) == y_).sum() / X_prob.shape[0]).item()
    #     # print(train_acc)
    #     X_probs.append(X_prob.detach().cpu().numpy()[np.arange(n_samples), ys].reshape(-1, 1))

    # print(f"Training accuracy {train_acc:.4f}")
    # res = np.hstack(X_probs)

    mlp_skorch = NeuralNetClassifier(MLP(9, 128, 2), callbacks="disable")
    num_crossval_folds = 5  # for efficiency; values like 5 or 10 will generally work better
    pred_probs = cross_val_predict(
        mlp_skorch,
        StandardScaler().fit_transform(Xs).astype("float32"),
        ys,
        cv=num_crossval_folds,
        method="predict_proba",
        n_jobs=-1,
    )

    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(ys, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f}")
    old_job_acc.append(acc)

    ranked_label_issues = find_label_issues(
        ys,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    # print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")
    ill_labels.append(len(ranked_label_issues))

    # Extract the cleaned data
    cleaned_Xs = np.delete(Xs, ranked_label_issues, axis=0)
    cleaned_ys = np.delete(ys, ranked_label_issues, axis=0)

    # # normalization
    # X_ = StandardScaler().fit_transform(cleaned_Xs)

    # X_ = torch.tensor(X_, dtype=torch.float32)
    # y_ = torch.tensor(cleaned_ys, dtype=torch.long)

    # # prob_i per epoch
    # X_probs = []
    # n_samples = X_.shape[0]

    # for e in range(200):
    #     optimizer.zero_grad()
    #     X_output = mlp(X_)
    #     X_prob = F.softmax(X_output, dim=1)
    #     loss = F.cross_entropy(X_prob, y_)
    #     loss.backward()
    #     optimizer.step()
    #     train_acc = ((X_prob.argmax(1) == y_).sum() / X_prob.shape[0]).item()
    #     # print(train_acc)
    #     X_probs.append(X_prob.detach().cpu().numpy()[np.arange(n_samples), cleaned_ys].reshape(-1, 1))

    # print(train_acc)

    pred_probs = cross_val_predict(
        mlp_skorch,
        StandardScaler().fit_transform(cleaned_Xs).astype("float32"),
        cleaned_ys,
        cv=num_crossval_folds,
        method="predict_proba",
        n_jobs=-1,
    )

    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(cleaned_ys, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f}")
    new_job_acc.append(acc)

old_job_acc = np.array(old_job_acc)
new_job_acc = np.array(new_job_acc)
ill_labels = np.array(ill_labels)

print(old_job_acc.mean(), old_job_acc.std())
print(new_job_acc.mean(), new_job_acc.std())
print(ill_labels.mean(), ill_labels.std())

# %%
sns.scatterplot(x=np.arange(len(old_job_acc)), y=old_job_acc, c="b", label="before")
sns.scatterplot(x=np.arange(len(new_job_acc)), y=new_job_acc, c="r", label="after")
for i in range(len(old_job_acc)):
    plt.plot([i, i], [old_job_acc[i], new_job_acc[i]], "--", c="k")
plt.title(fr"Acc before clean {old_job_acc.mean():.3f} $\pm$ {old_job_acc.std():.3f}" + "\n" +
          fr"after clean {new_job_acc.mean():.3f} $\pm$ {new_job_acc.std():.3f}" + "\n" +
          fr"cleaned samples {ill_labels.mean():.3f} $\pm$ {ill_labels.std():.3f}")
# plt.plot([], c="r", label="after")
# plt.plot([], c="b", label="before")
plt.plot([], "k--", label="improvement")
plt.legend()
plt.xlabel("Job ID")
plt.ylabel("Cross-Validate Accuracy")
plt.tight_layout()
# %%
