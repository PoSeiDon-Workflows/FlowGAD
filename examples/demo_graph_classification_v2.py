""" Example of graph classification problem

NOTE: with reduced anomaly data in training
"""


import sys 
import os, sys
sys.path.append('/Users/krishnanraghavan/Documents/Projects/Poseidon/graph_nn_2')



import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
from psd_gnn.dataset import Merge_PSD_Dataset, PSD_Dataset

from psd_gnn.models.graph_classifier import GNN
from psd_gnn.utils import process_args
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm




def train(model, loader):
    """ Train function

    Args:
        model (object): GNN model instance.
        loader (pyg.loader.DataLoader): Data loader object.

    Returns:
        float: Training accuracy.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    """ Evaluation function.

    Args:
        model (object): GNN model instance.
        loader (pyg.loader.DataLoader): Data loader object.

    Returns:
        tuple (float, list): Testing accuracy, predicted labels.
    """
    model.eval()

    total_correct = 0
    y_pred = []
    for data in loader:
        data = data.to(DEVICE)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        # aggregate the y_pred from batches
        y_pred += pred.detach().cpu().numpy().tolist()
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset), y_pred


if __name__ == "__main__":
    args = process_args()
    if args['workflow'] == "1000genome_new_2022":
        from psd_gnn.dataset_v2 import PSD_Dataset

    if args["seed"] != -1:
        random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])

    if args['gpu'] == -1:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args['gpu']}") if torch.cuda.is_available() else "cpu"

    if args['workflow'] == "all":
        dataset = Merge_PSD_Dataset(node_level=False, binary_labels=args['binary']).shuffle()
    else:
        dataset = PSD_Dataset("./", args['workflow'],
                              force_reprocess=args['force'],
                              node_level=False,
                              binary_labels=args['binary'],
                              anomaly_cat=args['anomaly_cat'],
                              anomaly_level=args['anomaly_level']
                              ).shuffle()
    # NOTE: imbalanced data in training
    idx = np.arange(len(dataset))
    Xs, ys = [], []
    for data in dataset:
        Xs.append(data.x.numpy())
        ys.append(data.y.item())
    Xs, ys = np.array(Xs), np.array(ys)

    ys_0 = idx[ys ==0]
    ys_1 = idx[ys > 0]

    train_0, test_0 = train_test_split(ys_0, train_size=0.6, random_state=0)
    val_0, test_0 = train_test_split(test_0, train_size=0.5, random_state=0)
    train_1, test_1 = train_test_split(ys_1, train_size=0.05, random_state=0)
    val_1, test_1 = train_test_split(test_1, train_size=0.5, random_state=0)

    train_idx = np.concatenate([train_0, train_1])
    val_idx = np.concatenate([val_0, val_1])
    test_idx = np.concatenate([test_0, test_1])

    n_graphs = len(dataset)
    # y = dataset.data.y.numpy()
    # train_idx, test_idx = train_test_split(
    #     np.arange(n_graphs), train_size=args['train_size'], random_state=0, shuffle=True)
    # val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0, shuffle=True)

    train_loader = DataLoader(dataset[train_idx], batch_size=args['batch_size'])
    val_loader = DataLoader(dataset[val_idx], batch_size=args['batch_size'])
    test_loader = DataLoader(dataset[test_idx], batch_size=args['batch_size'])

    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes

    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES, args['hidden_size'], NUM_OUT_FEATURES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # NOTE: weighted cross entropy loss
    class_weights = 1. / dataset.data.y.bincount()
    loss_func = CrossEntropyLoss(class_weights)

    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"{args['logdir']}/{args['workflow']}_{ts_start}")

    pbar = tqdm(range(args['epoch']), desc=f"{args['workflow']}")
    best = 0
    for e in pbar:
        model.train()
        optimizer.zero_grad()
        train_loss = train(model, train_loader)

        train_acc, _ = test(model, train_loader)

        val_acc, _ = test(model, val_loader)

        if val_acc > best:
            torch.save(model, osp.join("saved_models"))
        if args['verbose']:
            pbar.set_postfix({"train_loss": train_loss,
                              "train_acc": train_acc,
                              "val_acc": val_acc})
            # print(f"epoch {e:03d}",
            #       f"train loss {train_loss:.4f}",
            #       f"train acc {train_acc:.4f}",
            #       f"val acc {val_acc:.4f}",
            #       )
        writer.add_scalar("Loss", train_loss, e)
        writer.add_scalars("Accuracy", {"training": train_acc, "validation": val_acc}, e)
        # writer.add_scalar("Accuracy", train_acc, e)

    test_acc, y_pred = test(model, test_loader)

    y_true = []
    for data in test_loader:
        y_true += data.y.detach().cpu().numpy().tolist()

    if args['binary']:
        prec_val = precision_score(y_true, y_pred)
        f1_val = f1_score(y_true, y_pred)
        recall_val = recall_score(y_true, y_pred)
    else:
        conf_mat = confusion_matrix(y_true, y_pred)
        prec_val = precision_score(y_true, y_pred, average="weighted")
        f1_val = f1_score(y_true, y_pred, average="weighted")
        recall_val = recall_score(y_true, y_pred, average="weighted")

    print("graph level clf:",
          f"workflow {args['workflow']}",
          f"binary {args['binary']}",
          f"test acc {test_acc:.4f}",
          f"f1 {f1_val:.4f}",
          f"recall {recall_val:.4f}",
          f"prec {prec_val:.4f}",
          )

    # writer.add_hparams(args, {'acc': test_acc,
    #                           'f1': f1_val,
    #                           'recall': recall_val,
    #                           'prec': prec_val})
    writer.flush()
    writer.close()
