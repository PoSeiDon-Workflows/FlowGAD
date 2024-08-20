""" Example of graph classification problem """
import os.path as osp
import random
from datetime import datetime

import numpy as np
import torch
from psd_gnn.dataset import Merge_PSD_Dataset, PSD_Dataset

from psd_gnn.models.graph_classifier import GNN
from psd_gnn.utils import process_args, eval_metrics, create_dir
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader, ImbalancedSampler, NeighborLoader
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
    if args['workflow'] in ["1000genome_new_2022", "montage"]:
        from psd_gnn.dataset import PSD_Dataset

    if args["seed"] != -1:
        random.seed(args['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args['seed'])
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])

    if args['gpu'] == -1:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args['gpu']}") if torch.cuda.is_available() else "cpu"

    ROOT = osp.join("/tmp", "data", "psd", args['workflow'])
    if args['workflow'] == "all":
        dataset = Merge_PSD_Dataset(node_level=False, binary_labels=args['binary']).shuffle()
    else:
        dataset = PSD_Dataset(ROOT, args['workflow'],
                              force_reprocess=args['force'],
                              node_level=False,
                              binary_labels=args['binary'],
                              anomaly_cat=args['anomaly_cat'],
                              anomaly_level=args['anomaly_level'],
                              anomaly_num=args['anomaly_num'],
                              ).shuffle()

    n_graphs = len(dataset)
    # ys = dataset.data.y.numpy()

    # print(dataset.data.y.bincount())
    # print(dataset[0].edge_index.shape, dataset[0].x.shape)
    # exit()
    # split train/val/test
    train_idx, test_idx = train_test_split(
        np.arange(n_graphs), train_size=args['train_size'], random_state=0, shuffle=True)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0, shuffle=True)

    # REVIEW: imbalanced sampler
    # train_sampler = ImbalancedSampler(dataset[train_idx])
    # val_sampler = ImbalancedSampler(val_idx)
    # test_sampler = ImbalancedSampler(test_idx)
    # train_loader = DataLoader(dataset[train_idx], batch_size=args['batch_size'], sampler=train_sampler)
    # val_loader = DataLoader(dataset[val_idx], batch_size=args['batch_size'])
    # test_loader = DataLoader(dataset[test_idx], batch_size=args['batch_size'])

    train_loader = DataLoader(dataset[train_idx], batch_size=args['batch_size'])
    val_loader = DataLoader(dataset[val_idx], batch_size=args['batch_size'])
    test_loader = DataLoader(dataset[test_idx], batch_size=args['batch_size'])

    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes

    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES, args['hidden_size'], NUM_OUT_FEATURES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    # weight for imbalanced data
    if args['balance']:
        class_weight = 1 - dataset.data.y[train_idx].bincount() / dataset.data.y[train_idx].shape[0]
        loss_func = CrossEntropyLoss(weight=class_weight.to(DEVICE))
    else:
        loss_func = CrossEntropyLoss()

    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args['log']:
        writer = SummaryWriter(log_dir=f"{args['logdir']}/{args['workflow']}_{ts_start}")

    pbar = tqdm(range(args['epoch']), desc=f"{args['workflow']}", leave=True)
    best = 0
    for e in pbar:
        model.train()
        optimizer.zero_grad()
        train_loss = train(model, train_loader)

        train_acc, y_pred = test(model, train_loader)

        val_acc, _ = test(model, val_loader)

        if val_acc > best:
            # save tmp model on disk
            create_dir("tmp_models")
            torch.save(model, osp.join(f"tmp_models",
                                       f"saved_models_{args['workflow']}_{ts_start}"))
        if args['verbose']:
            pbar.set_postfix({"train_loss": train_loss,
                              "train_acc": train_acc,
                              "val_acc": val_acc})
        if args['log']:
            writer.add_scalar("Loss", train_loss, e)
            writer.add_scalars("Accuracy", {"training": train_acc,
                                            "validation": val_acc}, e)

    if args['verbose']:
        train_y_true = []
        for data in train_loader:
            train_y_true += data.y.detach().cpu().numpy().tolist()

        train_acc, train_y_pred = test(model, train_loader)
        train_metrics = eval_metrics(train_y_true, train_y_pred)
        print(f"Training acc {train_metrics['acc']:.4f}",
              f"roc-auc {train_metrics['roc_auc']:.4f}",
              f"f1 {train_metrics['f1']:.4f}",
              f"recall {train_metrics['recall']:.4f}",
              f"prec {train_metrics['prec']:.4f}")
        print(train_metrics['conf_mat'])

    test_acc, test_y_pred = test(model, test_loader)

    if args['verbose']:
        test_y_true = []
        for data in test_loader:
            test_y_true += data.y.detach().cpu().numpy().tolist()

        test_metrics = eval_metrics(test_y_true, test_y_pred)
        test_metrics = eval_metrics(test_y_true, test_y_pred)
        print(f"Testing acc {test_metrics['acc']:.4f}",
              f"roc-auc {test_metrics['roc_auc']:.4f}",
              f"f1 {test_metrics['f1']:.4f}",
              f"recall {test_metrics['recall']:.4f}",
              f"prec {test_metrics['prec']:.4f}",
              )
        print(test_metrics['conf_mat'])
    else:
        print(f"Testing acc {test_acc:.4f}")

    if args['log']:
        writer.flush()
        writer.close()
