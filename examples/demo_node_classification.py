""" Demo of node classification """

import random
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from psd_gnn.dataset import Merge_PSD_Dataset, PSD_Dataset
from psd_gnn.models.node_classifier import GNN
from psd_gnn.utils import eval_metrics, process_args


if __name__ == "__main__":

    args = process_args()
    if args["workflow"] in ["1000genome_new_2022", "montage"]:
        from psd_gnn.dataset_v2 import PSD_Dataset

    if args['seed'] != -1:
        torch.manual_seed(args['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args['seed'])
        np.random.seed(args['seed'])
        random.seed(args['seed'])

    if args['gpu'] == -1:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args['gpu']}") if torch.cuda.is_available() else "cpu"

    if args['workflow'] == "all":
        dataset = Merge_PSD_Dataset(node_level=True, binary_labels=args['binary'])
    else:
        dataset = PSD_Dataset("./", args['workflow'],
                              force_reprocess=args['force'],
                              node_level=True,
                              binary_labels=args['binary'],
                              anomaly_cat=args['anomaly_cat'],
                              anomaly_level=args['anomaly_level'],
                              anomaly_num=args['anomaly_num'],
                              )

    data = dataset[0]
    n_nodes = data.num_nodes

    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes

    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES,
                args['hidden_size'],
                NUM_OUT_FEATURES,
                n_conv_blocks=args['conv_blocks'],
                dropout=args['dropout']).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    # weights for imbalanced data
    class_weight = 1 - data.y[data.train_mask].bincount() / data.y[data.train_mask].shape[0]
    loss_func = CrossEntropyLoss(weight=class_weight.to(DEVICE))

    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args['log']:
        writer = SummaryWriter(log_dir=f"{args['logdir']}/{args['workflow']}_{ts_start}")
    data = data.to(DEVICE)

    pbar = tqdm(range(args['epoch']), desc=args['workflow'], leave=True)
    model.train()
    for e in pbar:
        optimizer.zero_grad()
        y_hat = model(data.x, data.edge_index)
        train_loss = loss_func(y_hat[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        train_y_true = data.y[data.train_mask].detach().cpu().numpy()
        train_y_pred = y_hat[data.train_mask].argmax(dim=1).detach().cpu().numpy()
        train_acc = eval_metrics(train_y_true, train_y_pred, metric="acc")

        val_loss = loss_func(y_hat[data.val_mask], data.y[data.val_mask])
        val_y_true = data.y[data.val_mask].detach().cpu().numpy()
        val_y_pred = y_hat[data.val_mask].argmax(dim=1).detach().cpu().numpy()
        val_acc = eval_metrics(val_y_true, val_y_pred, metric="acc")

        if args['verbose']:
            pbar.set_postfix({"train_loss": train_loss.detach().cpu().item(),
                             "train_acc": train_acc,
                              "val_acc": val_acc})

        if args['log']:
            writer.add_scalar("Loss", train_loss, e)
            writer.add_scalars("Accuracy", {"Training": train_acc,
                                            "Validation": val_acc}, e)

    if args['verbose']:
        train_metrics = eval_metrics(train_y_true, train_y_pred)
        print(f"Training acc {train_metrics['acc']:.4f}",
              f"roc-auc {train_metrics['roc_auc']:.4f}",
              f"f1 {train_metrics['f1']:.4f}",
              f"recall {train_metrics['recall']:.4f}",
              f"prec {train_metrics['prec']:.4f}")
        print(train_metrics['conf_mat'])

    test_y_true = data.y[data.test_mask].detach().cpu().numpy()
    test_y_pred = y_hat[data.test_mask].argmax(dim=1).detach().cpu().numpy()

    if args['verbose']:
        test_metrics = eval_metrics(test_y_true, test_y_pred)
        print(f"Testing acc {test_metrics['acc']:.4f}",
              f"roc-auc {test_metrics['roc_auc']:.4f}",
              f"f1 {test_metrics['f1']:.4f}",
              f"recall {test_metrics['recall']:.4f}",
              f"prec {test_metrics['prec']:.4f}",
              )
        print(test_metrics['conf_mat'])
    else:
        test_acc = eval_metrics(test_y_true, test_y_pred, metric="acc")
        print(f"Testing acc {test_acc:.4f}")

    if args['log']:
        writer.flush()
        writer.close()
