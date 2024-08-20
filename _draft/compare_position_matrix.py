""" Compare models w/ and w/o position matrix

TODO: 
* HPS:
* clean data - HPS

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

# %% [define the training function]


def train(model, data, name="MLP", pos=False, **kwargs):
    """ Train the model

    Args:
        model (Module): Model to train.
        data (pyg.Data): Data object.
        name (str, optional): Description of the model. Defaults to "MLP".
        pos (bool, optional): Use position matrix if `True`. Defaults to False.
    """
    optimizer = Adam(model.parameters(), lr=kwargs.get("lr", 1e-3))
    model.reset_parameters()

    x = data.x if not pos else data.x_aug
    desc = f"{name} w/o P" if not pos else f"{name} w/  P"
    pbar = tqdm(range(1000), desc=desc, leave=True)

    model.train()
    patient = 10
    best = float("inf")
    for e in pbar:
        optimizer.zero_grad()
        out = model(x, data.edge_index)
        train_out = out[data.train_mask]
        loss = F.cross_entropy(train_out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_out = out[data.val_mask]
        val_loss = F.cross_entropy(val_out, data.y[data.val_mask])

        pbar.set_postfix({"train_loss": loss.item(),
                          "val_loss": val_loss.item()})

        if val_loss <= best:
            best = val_loss
            patient = 100
        else:
            patient -= 1
        if patient <= 0:
            print(f"Early stopping at epoch {e}")
            break


def test(model, data, pos=False, **kwargs):
    """ Eval the model performance on test set.

    Args:
        model (Module): Model to eval.
        data (pyg.Data): Data object.
        pos (bool, optional): Use position matrix if `True`. Defaults to False.

    Returns:
        dict: A dictionary of results including accuracy, precision, recall, f1, auc.
    """
    model.eval()
    x = data.x if not pos else data.x_aug
    out = model(x, data.edge_index)
    test_pred = out[data.test_mask].argmax(1).cpu().detach().numpy()
    test_y = data.y[data.test_mask].cpu().detach().numpy()
    res = eval_metrics(test_y, test_pred)
    return res


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
data.x_aug = torch.cat([data.x, torch.eye(ds_node.num_nodes_per_graph, device=DEVICE).repeat(n_graphs, 1)], dim=1)

# %% [test models]

model_args = {"in_channels": ds_node.num_node_features,
              "hidden_channels": 128,
              "num_layers": 3,
              "out_channels": ds_node.num_classes,
              "dropout": 0.5}
model_args_aug = {"in_channels": data.x_aug.shape[1],
                  "hidden_channels": 128,
                  "num_layers": 3,
                  "out_channels": ds_node.num_classes,
                  "dropout": 0.5}

mlp_model = MLP(**model_args).to(DEVICE)
gcn_model = GCN(**model_args).to(DEVICE)
gat_model = GAT(**model_args).to(DEVICE)
graphsage_model = GraphSAGE(**model_args).to(DEVICE)

mlp_model_aug = MLP(**model_args_aug).to(DEVICE)
gcn_model_aug = GCN(**model_args_aug).to(DEVICE)
gat_model_aug = GAT(**model_args_aug).to(DEVICE)
graphsage_model_aug = GraphSAGE(**model_args_aug).to(DEVICE)

# %% [Train and eval models]

train(mlp_model, data, name="MLP", pos=False)
mlp_res = test(mlp_model, data, pos=False)
train(mlp_model_aug, data, name="MLP", pos=True)
mlp_res_aug = test(mlp_model_aug, data, pos=True)

train(gcn_model, data, name="GCN", pos=False)
gcn_res = test(gcn_model, data, pos=False)
train(gcn_model_aug, data, name="GCN", pos=True)
gcn_res_aug = test(gcn_model_aug, data, pos=True)

train(gat_model, data, name="GAT", pos=False)
gat_res = test(gat_model, data, pos=False)
train(gat_model_aug, data, name="GAT", pos=True)
gat_res_aug = test(gat_model_aug, data, pos=True)

train(graphsage_model, data, name="GraphSAGE", pos=False)
graphsage_res = test(graphsage_model, data, pos=False)
train(graphsage_model_aug, data, name="GraphSAGE", pos=True)
graphsage_res_aug = test(graphsage_model_aug, data, pos=True)

print("MLP w/o P", mlp_res)
print("MLP w/  P", mlp_res_aug)
print("GCN w/o P", gcn_res)
print("GCN w/  P", gcn_res_aug)
print("GAT w/o P", gat_res)
print("GAT w/  P", gat_res_aug)
print("GraphSAGE w/o P", graphsage_res)
print("GraphSAGE w/  P", graphsage_res_aug)
