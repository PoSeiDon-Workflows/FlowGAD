""" Demo of node classification """

from pydoc import describe
import numpy as np
import random
import torch
from psd_gnn.dataset import Merge_PSD_Dataset, PSD_Dataset
from psd_gnn.models.node_classifier import GNN
from psd_gnn.utils import process_args
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    args = process_args()
    if args["workflow"] == "1000genome_new_2022":
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
                              )

    data = dataset[0]
    n_nodes = data.num_nodes

    idx = np.arange(n_nodes)
    ys = data.y.numpy()
    ys_0 = idx[ys == 0]
    ys_1 = idx[ys > 0]
    train_0, test_0 = train_test_split(ys_0, train_size=0.6, random_state=0)
    val_0, test_0 = train_test_split(test_0, train_size=0.5, random_state=0)
    train_1, test_1 = train_test_split(ys_1, train_size=0.05, random_state=0)
    val_1, test_1 = train_test_split(test_1, train_size=0.5, random_state=0)
    
    train_idx = np.concatenate([train_0, train_1])
    val_idx = np.concatenate([val_0, val_1])
    test_idx = np.concatenate([test_0, test_1])

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = 1

    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[val_idx] = 1

    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = 1
    # NOTE: reprocess the train_mask, val_mask and test_mask
    
    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes

    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES, args['hidden_size'], NUM_OUT_FEATURES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    class_weights = 1. / data.y.bincount()
    loss_func = CrossEntropyLoss(class_weights)

    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        val_acc = accuracy_score(data.y[data.val_mask].detach().cpu().numpy(),
                                 y_hat[data.val_mask].argmax(dim=1).detach().cpu().numpy())
        train_loss.backward()
        optimizer.step()

        if args['verbose']:
            pbar.set_postfix({"train_loss": train_loss.detach().cpu().item(),
                             "train_acc": train_acc,
                              "val_acc": val_acc})
            # print(f"epoch {e:03d}",
            #       f"train loss {train_loss:.4f}",
            #       f"train acc {train_acc:.4f}",
            #       f"val acc {val_acc:.4f}",
            #       )
        writer.add_scalar("Loss", train_loss, e)
        writer.add_scalars("Accuracy", {"Training": train_acc,
                                        "Validation": val_acc}, e)
    model.eval()
    y_out = model(data.x, data.edge_index)
    y_true = data.y[data.test_mask].detach().cpu().numpy()
    y_pred = y_out[data.test_mask].argmax(dim=1).detach().cpu().numpy()
    test_acc = accuracy_score(y_true, y_pred)
    if args['binary']:
        test_prec = precision_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
    else:
        conf_mat = confusion_matrix(y_true, y_pred)
        test_prec = precision_score(y_true, y_pred, average="weighted")
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        test_recall = recall_score(y_true, y_pred, average="weighted")

    print("node level clf:",
          f"workflow {args['workflow']}",
          f"binary {args['binary']}",
          f"test acc {test_acc:.4f}",
          f"f1 {test_f1:.4f}",
          f"recall {test_recall:.4f}",
          f"prec {test_prec:.4f}",
          )

    # writer.add_hparams(args, {'acc': test_acc,
    #                           'f1': test_f1,
    #                           'recall': test_recall,
    #                           'prec': test_prec})
    writer.flush()
    writer.close()
