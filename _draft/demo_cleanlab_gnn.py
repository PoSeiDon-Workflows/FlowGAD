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
                             roc_auc_score, f1_score)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from torch.nn import CrossEntropyLoss, Linear, Module, ReLU
from torch.optim import Adam

from psd_gnn.utils import parse_adj
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.models.node_classifier import GNN

# %%
ts_features = ["ready", "submit", "execute_start", "execute_end", "post_script_start", "post_script_end"]
delay_features = ["wms_delay", "queue_delay", "runtime", "post_script_delay", "stage_in_delay", "stage_out_delay"]

np.random.seed(41)
torch.manual_seed(41)

# %%

dataset = PSD_Dataset(name="1000genome_new_2022",
                      node_level=True,
                      # force_reprocess=True,
                      binary_labels=True,
                      normalize=False)
NUM_NODE_FEATURES = dataset.num_node_features
NUM_OUT_FEATURES = dataset.num_classes
data = dataset[0]
model = GNN(NUM_NODE_FEATURES,
            128,
            NUM_OUT_FEATURES,
            n_conv_blocks=2,
            dropout=0.5)
model = MLPClassifier((128, 128, 128))
mlp_skorch = NeuralNetClassifier(model, callbacks="disable")
num_crossval_folds = 5  # for efficiency; values like 5 or 10 will generally work better
Xs = dataset.data.x.numpy()
ys = dataset.data.y.numpy()
pred_probs = cross_val_predict(
    # mlp_skorch,
    model,
    StandardScaler().fit_transform(Xs),
    ys,
    cv=num_crossval_folds,
    method="predict_proba",
    n_jobs=-1,
)

predicted_labels = pred_probs.argmax(axis=1)
acc = accuracy_score(ys, predicted_labels)
acc = accuracy_score(ys, predicted_labels)
f1 = f1_score(ys, predicted_labels)
roc_auc = roc_auc_score(ys, predicted_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f} {f1:.4f} {roc_auc:.4f}")


# %%

ranked_label_issues = find_label_issues(
    ys,
    pred_probs,
    return_indices_ranked_by="self_confidence",
)

print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
# print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")
# ill_labels.append(len(ranked_label_issues))

# Extract the cleaned data
cleaned_Xs = np.delete(Xs, ranked_label_issues, axis=0)
cleaned_ys = np.delete(ys, ranked_label_issues, axis=0)

# %%

pred_probs = cross_val_predict(
    model,
    StandardScaler().fit_transform(cleaned_Xs).astype("float32"),
    cleaned_ys,
    cv=num_crossval_folds,
    method="predict_proba",
    n_jobs=-1,
)

predicted_labels = pred_probs.argmax(axis=1)
acc = accuracy_score(cleaned_ys, predicted_labels)
print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f}")


# %%


for top_idx in np.arange(1000, len(ranked_label_issues), 1000):
    cleaned_Xs = np.delete(Xs, ranked_label_issues[:top_idx], axis=0)
    cleaned_ys = np.delete(ys, ranked_label_issues[:top_idx], axis=0)
    pred_probs = cross_val_predict(
        model,
        StandardScaler().fit_transform(cleaned_Xs).astype("float32"),
        cleaned_ys,
        cv=num_crossval_folds,
        method="predict_proba",
        # n_jobs=-1,
    )
    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(cleaned_ys, predicted_labels)
    f1 = f1_score(cleaned_ys, predicted_labels)
    roc_auc = roc_auc_score(cleaned_ys, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {acc:.4f} {f1:.4f} {roc_auc:.4f}")

# %%


# %%
# sns.scatterplot(x=np.arange(len(old_job_acc)), y=old_job_acc, c="b", label="before")
# sns.scatterplot(x=np.arange(len(new_job_acc)), y=new_job_acc, c="r", label="after")
# for i in range(len(old_job_acc)):
#     plt.plot([i, i], [old_job_acc[i], new_job_acc[i]], "--", c="k")
# plt.title(fr"Acc before clean {old_job_acc.mean():.3f} $\pm$ {old_job_acc.std():.3f}" + "\n" +
#           fr"after clean {new_job_acc.mean():.3f} $\pm$ {new_job_acc.std():.3f}" + "\n" +
#           fr"cleaned samples {ill_labels.mean():.3f} $\pm$ {ill_labels.std():.3f}")
# # plt.plot([], c="r", label="after")
# # plt.plot([], c="b", label="before")
# plt.plot([], "k--", label="improvement")
# plt.legend()
# plt.xlabel("Job ID")
# plt.ylabel("Cross-Validate Accuracy")
# plt.tight_layout()
res = np.array([[0.7435, 0.6105, 0.7109],
                [0.7723, 0.6414, 0.7351],
                [0.7816, 0.6572, 0.7478],
                [0.8064, 0.6940, 0.7758],
                [0.8244, 0.7142, 0.7923],
                [0.8280, 0.7108, 0.7911],
                [0.8511, 0.7515, 0.8246],
                [0.8825, 0.8029, 0.8656],
                [0.8863, 0.8096, 0.8675],
                [0.9190, 0.8679, 0.9082]])

sns.lineplot(x=np.arange(1000, len(ranked_label_issues), 1000), y=res[:, 0], c="r", label="Accuracy")
sns.lineplot(x=np.arange(1000, len(ranked_label_issues), 1000), y=res[:, 1], c="b", label="F1")
sns.lineplot(x=np.arange(1000, len(ranked_label_issues), 1000), y=res[:, 2], c="g", label="ROC-AUC")
plt.legend()
plt.xlabel("# of Top ill samples")
# %%
