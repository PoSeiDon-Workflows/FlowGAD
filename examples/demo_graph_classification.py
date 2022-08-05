
import random

import numpy as np
import torch
from psd_gnn.dataset import Merge_PSD_Dataset, PSD_Dataset
from psd_gnn.models.graph_classifier import GNN, GNN_v2
from psd_gnn.utils import process_args
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def train(model, loader):

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
                              force_reprocess=True,
                              node_level=False,
                              binary_labels=args['binary'],
                              anomaly_cat=args['anomaly_cat'],
                              anomaly_level=args['anomaly_level']
                              ).shuffle()

    n_graphs = len(dataset)
    y = dataset.data.y.numpy()
    train_idx, test_idx = train_test_split(np.arange(n_graphs), train_size=args['train_size'], random_state=0, shuffle=True)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0, shuffle=True)

    train_loader = DataLoader(dataset[train_idx], batch_size=args['batch_size'])
    val_loader = DataLoader(dataset[val_idx], batch_size=args['batch_size'])
    test_loader = DataLoader(dataset[test_idx], batch_size=args['batch_size'])

    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes

    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES, args['hidden_size'], NUM_OUT_FEATURES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_func = CrossEntropyLoss()

    pbar = tqdm(range(args['epoch']), desc=f"{args['workflow']}")
    for e in pbar:
        model.train()
        optimizer.zero_grad()
        train_loss = train(model, train_loader)

        train_acc, _ = test(model, train_loader)

        val_acc, _ = test(model, val_loader)

        if args['verbose']:
            print(f"epoch {e:03d}",
                  f"train loss {train_loss:.4f}",
                  f"train acc {train_acc:.4f}",
                  f"val acc {val_acc:.4f}",
                  )

    test_acc, y_pred = test(model, test_loader)

    y_true = []
    for data in test_loader:
        y_true += data.y.detach().cpu().numpy().tolist()

    if args['binary']:
        prec_val = precision_score(y_true, y_pred)
        f1_val = f1_score(y_true, y_pred)
        recall_val = recall_score(y_true, y_pred)
    else:
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
