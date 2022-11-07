""" Demo of node classification """

import numpy as np
import random
import torch
from psd_gnn.dataset import Merge_PSD_Dataset, PSD_Dataset
from psd_gnn.models.node_classifier import GNN
from psd_gnn.utils import process_args
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    args = process_args()
    if args["workflow"] in ["1000genome_new_2022", "montage"]:
        from psd_gnn.dataset_v2 import PSD_Dataset

    if args['seed'] != -1:
        torch.manual_seed(args['seed'])
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
                NUM_OUT_FEATURES, dropout=args['dropout']).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    loss_func = CrossEntropyLoss()

    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args['log']:
        writer = SummaryWriter(log_dir=f"{args['logdir']}/{args['workflow']}_{ts_start}")
    data = data.to(DEVICE)
    pbar = tqdm(range(args['epoch']), desc=args['workflow'])
    for e in pbar:
        model.train()
        optimizer.zero_grad()
        y_hat = model(data.x, data.edge_index)

        train_loss = loss_func(y_hat[data.train_mask], data.y[data.train_mask])

        train_acc = accuracy_score(data.y[data.train_mask].detach().cpu().numpy(),
                                   y_hat[data.train_mask].argmax(dim=1).detach().cpu().numpy())
        train_loss.backward()
        optimizer.step()

        val_y_true = data.y[data.val_mask].detach().cpu().numpy()
        val_y_pred = y_hat[data.val_mask].argmax(dim=1).detach().cpu().numpy()
        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_prec = precision_score(val_y_true, val_y_pred)
        val_f1 = f1_score(val_y_true, val_y_pred)
        val_recall = recall_score(val_y_true, val_y_pred)
        val_roc_auc = roc_auc_score(val_y_true, val_y_pred)

        if args['verbose']:
            pbar.set_postfix({"train_loss": train_loss.detach().cpu().item(),
                             "train_acc": train_acc,
                              "val_acc": val_acc})
            # print(f"epoch {e:03d}",
            #       f"train loss {train_loss:.4f}",
            #       f"train acc {train_acc:.4f}",
            #       f"val acc {val_acc:.4f}",
            #       )
        if args['log']:
            writer.add_scalar("Loss", train_loss, e)
            writer.add_scalars("Accuracy", {"Training": train_acc,
                                            "Validation": val_acc}, e)
    y_true = data.y[data.train_mask].detach().cpu().numpy()
    y_pred = y_hat[data.train_mask].argmax(dim=1).detach().cpu().numpy()
    print("training", confusion_matrix(y_true, y_pred))

    model.eval()
    y_out = model(data.x, data.edge_index)
    y_true = data.y[data.test_mask].detach().cpu().numpy()
    y_pred = y_out[data.test_mask].argmax(dim=1).detach().cpu().numpy()
    test_acc = accuracy_score(y_true, y_pred)
    if args['binary']:
        conf_mat = confusion_matrix(y_true, y_pred)
        test_prec = precision_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
        test_roc_auc = roc_auc_score(y_true, y_pred)
    else:
        conf_mat = confusion_matrix(y_true, y_pred)
        test_prec = precision_score(y_true, y_pred, average="weighted")
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        test_recall = recall_score(y_true, y_pred, average="weighted")
        test_roc_auc = roc_auc_score(y_true, y_pred, average="weighted")

    print("node level clf:",
          f"workflow {args['workflow']}",
          f"binary {args['binary']}",
          f"test acc {test_acc:.4f}",
          f"roc-auc {test_roc_auc:.4f}",
          f"f1 {test_f1:.4f}",
          f"recall {test_recall:.4f}",
          f"prec {test_prec:.4f}",
          )
    print("test", conf_mat)

    if args['log']:
        writer.flush()
        writer.close()
