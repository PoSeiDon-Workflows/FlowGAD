""" Compare models w/ and w/o position matrix

Results:
```
dataset: montage
GCN (X, A):
{'acc': 0.8000057977736549, 'f1': 0.736209878452537, 'prec': 0.7566714284395928, 'recall': 0.8000057977736549, 'roc_auc': 0.5337532076686248, 'conf_mat': array([[27002,   474],
       [ 6425,   595]])}
GCN (X+P, A):
{'acc': 0.8016581632653061, 'f1': 0.7406713047298731, 'prec': 0.7619680470327233, 'recall': 0.8016581632653061, 'roc_auc': 0.5394038578708836, 'conf_mat': array([[26972,   504],
       [ 6338,   682]])}
MLP (X):
{'acc': 0.8077458256029685, 'f1': 0.7480702124608598, 'prec': 0.7831350087006692, 'recall': 0.8077458256029685, 'roc_auc': 0.5479978382584293, 'conf_mat': array([[27092,   384],
       [ 6248,   772]])}
MLP (X+P):
{'acc': 0.7954255565862709, 'f1': 0.7644559898698505, 'prec': 0.7591027327369807, 'recall': 0.7954255565862709, 'roc_auc': 0.5845416709698263, 'conf_mat': array([[25832,  1644],
       [ 5413,  1607]])}
GAT (X, A):
{'acc': 0.7964981447124304, 'f1': 0.7062732532149596, 'prec': 0.6344092945303438, 'recall': 0.7964981447124304, 'roc_auc': 0.5, 'conf_mat': array([[27476,     0],
       [ 7020,     0]])}
GAT (X+P, A):
{'acc': 0.7964981447124304, 'f1': 0.7062732532149596, 'prec': 0.6344092945303438, 'recall': 0.7964981447124304, 'roc_auc': 0.5, 'conf_mat': array([[27476,     0],
       [ 7020,     0]])}
GraphSAGE (X, A):
{'acc': 0.8142973098330241, 'f1': 0.7718935129537141, 'prec': 0.7887422502044443, 'recall': 0.8142973098330241, 'roc_auc': 0.5832375854358676, 'conf_mat': array([[26731,   745],
       [ 5661,  1359]])}
GraphSAGE (X+P, A):
{'acc': 0.8092822356215214, 'f1': 0.7685177521517242, 'prec': 0.777995955884907, 'recall': 0.8092822356215214, 'roc_auc': 0.5806726844541664, 'conf_mat': array([[26547,   929],
       [ 5650,  1370]])}
```

* HPS:
* Overfitting:
* clean data - HPS
.. math::

f(x):
"""

# %% [imports]

import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import Adam
from torch_geometric.nn.models import GAT, GCN, MLP, GraphSAGE
from tqdm import tqdm

from psd_gnn.dataset import PSD_Dataset
from psd_gnn.transforms import MinMaxNormalizeFeatures
from psd_gnn.utils import eval_metrics

torch.manual_seed(0)
# %% [load data]

workflow = "montage"
ROOT = osp.join(osp.expanduser("~"), "tmp", "data", workflow)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pre_transform = T.Compose([MinMaxNormalizeFeatures(),
                           T.ToUndirected(),
                           T.RandomNodeSplit(split="train_rest",
                                             num_val=0.2,
                                             num_test=0.2)])
ds_node = PSD_Dataset(root=ROOT,
                      name=workflow,
                      binary_labels=True,
                      node_level=True,
                      normalize=False,
                      pre_transform=pre_transform,
                      force_reprocess=False)

data = ds_node[0].to(DEVICE)

# augment the feature with position matrix
n_graphs = data.x.shape[0] // ds_node.num_nodes_per_graph
data_x_aug = torch.cat([data.x, torch.eye(ds_node.num_nodes_per_graph, device=DEVICE).repeat(n_graphs, 1)], dim=1)

# %% [test models]

model_args = {"in_channels": ds_node.num_node_features,
              "hidden_channels": 128,
              "num_layers": 3,
              "out_channels": ds_node.num_classes,
              "dropout": 0.5}
model_args_aug = {"in_channels": data_x_aug.shape[1],
                  "hidden_channels": 128,
                  "num_layers": 3,
                  "out_channels": ds_node.num_classes,
                  "dropout": 0.5}

gcn_model = GCN(**model_args).to(DEVICE)
mlp_model = MLP(**model_args).to(DEVICE)
gat_model = GAT(**model_args).to(DEVICE)
graphsage_model = GraphSAGE(**model_args).to(DEVICE)

gcn_model_aug = GCN(**model_args_aug).to(DEVICE)
mlp_model_aug = MLP(**model_args_aug).to(DEVICE)
gat_model_aug = GAT(**model_args_aug).to(DEVICE)
graphsage_model_aug = GraphSAGE(**model_args_aug).to(DEVICE)

gcn_optim = Adam(gcn_model.parameters(), lr=0.001)
mlp_optim = Adam(mlp_model.parameters(), lr=0.001)
gat_optim = Adam(gat_model.parameters(), lr=0.001)
graphsage_optim = Adam(graphsage_model.parameters(), lr=0.001)

gcn_optim_aug = Adam(gcn_model_aug.parameters(), lr=0.001)
mlp_optim_aug = Adam(mlp_model_aug.parameters(), lr=0.001)
gat_optim_aug = Adam(gat_model_aug.parameters(), lr=0.001)
graphsage_optim_aug = Adam(graphsage_model_aug.parameters(), lr=0.001)

# %% [Train GCN]

gcn_model.train()
pbar = tqdm(range(1000), desc="GCN (X, A)", leave=True)
for e in pbar:
    gcn_optim.zero_grad()
    out = gcn_model(data.x, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    gcn_optim.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

gcn_model.eval()
test_out = gcn_model(data.x, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
gcn_res = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(gcn_res)

# augmented x
gcn_model_aug.train()
pbar = tqdm(range(1000), desc="GCN (X+P, A)", leave=True)
for e in pbar:
    gcn_optim_aug.zero_grad()
    out = gcn_model_aug(data_x_aug, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    gcn_optim_aug.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

gcn_model_aug.eval()
test_out = gcn_model_aug(data_x_aug, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
gcn_res_x_aug = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(gcn_res_x_aug)

# %% [Train MLP]

mlp_model.train()
pbar = tqdm(range(1000), desc="MLP (X)", leave=True)
for e in pbar:
    mlp_optim.zero_grad()
    out = mlp_model(data.x, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    mlp_optim.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

mlp_model.eval()
test_out = mlp_model(data.x, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
mlp_res = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(mlp_res)

# augmented x
mlp_model_aug.train()
pbar = tqdm(range(1000), desc="MLP (X+P)", leave=True)
for e in pbar:
    mlp_optim_aug.zero_grad()
    out = mlp_model_aug(data_x_aug, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    mlp_optim_aug.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

mlp_model_aug.eval()
test_out = mlp_model_aug(data_x_aug, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
mlp_res_x_aug = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(mlp_res_x_aug)

# %% [Train GAT]

gat_model.train()
pbar = tqdm(range(1000), desc="GAT (X, A)", leave=True)
for e in pbar:
    gat_optim.zero_grad()
    out = gat_model(data.x, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    gat_optim.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

gat_model.eval()
test_out = gat_model(data.x, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
gat_res = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(gat_res)

# augmented x
gat_model_aug.train()
pbar = tqdm(range(1000), desc="GAT (X+P, A)", leave=True)
for e in pbar:
    gat_optim_aug.zero_grad()
    out = gat_model_aug(data_x_aug, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    gat_optim_aug.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

gat_model_aug.eval()
test_out = gat_model_aug(data_x_aug, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
gat_res_x_aug = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(gat_res_x_aug)

# %% [Train GraphSAGE]

graphsage_model.train()
pbar = tqdm(range(1000), desc="GraphSAGE (X, A)", leave=True)
for e in pbar:
    graphsage_optim.zero_grad()
    out = graphsage_model(data.x, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    graphsage_optim.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

graphsage_model.eval()
test_out = graphsage_model(data.x, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
graphsage_res = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(graphsage_res)


# augmented x
graphsage_model_aug.train()
pbar = tqdm(range(1000), desc="GraphSAGE (X+P, A)", leave=True)
for e in pbar:
    graphsage_optim_aug.zero_grad()
    out = graphsage_model_aug(data_x_aug, data.edge_index)
    train_out = out[data.train_mask]
    train_loss = F.cross_entropy(train_out, data.y[data.train_mask])
    train_loss.backward()
    graphsage_optim_aug.step()

    val_out = out[data.val_mask]
    val_loss = F.cross_entropy(val_out, data.y[data.val_mask])
    pbar.set_postfix({"train loss": train_loss.item(),
                      "val loss": val_loss.item()})

graphsage_model_aug.eval()
test_out = graphsage_model_aug(data_x_aug, data.edge_index)[data.test_mask]
test_pred = test_out.argmax(dim=1).cpu().detach().numpy()
graphsage_res_x_aug = eval_metrics(data.y[data.test_mask].cpu().detach().numpy(), test_pred)
print(graphsage_res_x_aug)
