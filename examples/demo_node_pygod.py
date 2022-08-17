""" Demo of node classification for binary case """

from shutil import ExecError
import numpy as np
import random
import torch
from psd_gnn.dataset_v2 import Merge_PSD_Dataset, PSD_Dataset
from psd_gnn.models.node_classifier import GNN, GNN_v2
from psd_gnn.utils import process_args, init_model
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.nn import CrossEntropyLoss
from pygod.metrics import (eval_average_precision, eval_recall_at_k,
                           eval_roc_auc)
from pygod.models import *
from tqdm import tqdm
from random import choice

if __name__ == "__main__":

    args = process_args()

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

    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes
    epoch = 200
    hid_dim = [32, 64, 128, 256]
    weight_decay = 0.001
    dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lr = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    batch_size = 0
    gpu = 0
    num_neigh = -1
    alpha = [0.8, 0.5, 0.2]
    model_name = "adone"
    for model_name in ['mlpae', 'scan', 'radar', 'anomalous', 'gcnae', 'dominant', 'done', 'adone', 'gaan', 'guide', 'conad']:
        try:
            # model = init_model(args)
            if model_name == "adone":
                model = AdONE(hid_dim=choice(hid_dim),
                              weight_decay=weight_decay,
                              dropout=choice(dropout),
                              lr=choice(lr),
                              epoch=epoch,
                              gpu=gpu,
                              batch_size=batch_size,
                              num_neigh=num_neigh)
            elif model_name == 'anomalydae':
                hd = choice(hid_dim)
                model = AnomalyDAE(embed_dim=hd,
                                   out_dim=hd,
                                   weight_decay=weight_decay,
                                   dropout=choice(dropout),
                                   theta=choice([10., 40., 90.]),
                                   eta=choice([3., 5., 8.]),
                                   lr=choice(lr),
                                   epoch=epoch,
                                   gpu=gpu,
                                   alpha=choice(alpha),
                                   batch_size=batch_size,
                                   num_neigh=num_neigh)
            elif model_name == 'conad':
                model = CONAD(hid_dim=choice(hid_dim),
                              weight_decay=weight_decay,
                              dropout=choice(dropout),
                              lr=choice(lr),
                              epoch=epoch,
                              gpu=gpu,
                              alpha=choice(alpha),
                              batch_size=batch_size,
                              num_neigh=num_neigh)
            elif model_name == 'dominant':
                model = DOMINANT(hid_dim=choice(hid_dim),
                                 weight_decay=weight_decay,
                                 dropout=choice(dropout),
                                 lr=choice(lr),
                                 epoch=epoch,
                                 gpu=gpu,
                                 alpha=choice(alpha),
                                 batch_size=batch_size,
                                 num_neigh=num_neigh)
            elif model_name == 'done':
                model = DONE(hid_dim=choice(hid_dim),
                             weight_decay=weight_decay,
                             dropout=choice(dropout),
                             lr=choice(lr),
                             epoch=epoch,
                             gpu=gpu,
                             batch_size=batch_size,
                             num_neigh=num_neigh)
            elif model_name == 'gaan':
                model = GAAN(noise_dim=choice([8, 16, 32]),
                             hid_dim=choice(hid_dim),
                             weight_decay=weight_decay,
                             dropout=choice(dropout),
                             lr=choice(lr),
                             epoch=epoch,
                             gpu=gpu,
                             alpha=choice(alpha),
                             batch_size=batch_size,
                             num_neigh=num_neigh)
            elif model_name == 'gcnae':
                model = GCNAE(hid_dim=choice(hid_dim),
                              weight_decay=weight_decay,
                              dropout=choice(dropout),
                              lr=choice(lr),
                              epoch=epoch,
                              gpu=gpu,
                              batch_size=batch_size,
                              num_neigh=num_neigh)
            elif model_name == 'guide':
                model = GUIDE(a_hid=choice(hid_dim),
                              s_hid=choice([4, 5, 6]),
                              weight_decay=weight_decay,
                              dropout=choice(dropout),
                              lr=choice(lr),
                              epoch=epoch,
                              gpu=gpu,
                              alpha=choice(alpha),
                              batch_size=batch_size,
                              num_neigh=num_neigh,
                              cache_dir='./tmp')
            elif model_name == "mlpae":
                model = MLPAE(hid_dim=choice(hid_dim),
                              weight_decay=weight_decay,
                              dropout=choice(dropout),
                              lr=choice(lr),
                              epoch=epoch,
                              gpu=gpu,
                              batch_size=batch_size)
            # elif model_name == 'lof':
            #     model = LOF()
            # elif model_name == 'if':
            #     model = IsolationForest()
            elif model_name == 'radar':
                model = Radar(lr=choice(lr), gpu=gpu)
            elif model_name == 'anomalous':
                model = ANOMALOUS(lr=choice(lr), gpu=gpu)
            elif model_name == 'scan':
                model = SCAN(eps=choice([0.3, 0.5, 0.8]), mu=choice([2, 5, 10]))

            model.fit(data)
            score = model.decision_scores_
            roc_auc_val = eval_roc_auc(data.y.numpy(), score)
            ap_val = eval_average_precision(data.y.numpy(), score)
            rec_val = eval_recall_at_k(data.y.numpy(), score, data.y.numpy().sum())
            print(f"{model_name}",
                  f"ROC {roc_auc_val:.4f}",
                  f"AP {ap_val:.4f}",
                  f"Recall {rec_val:.4f}")
        except Exception:
            print(f"error in {model_name}")
    exit()
    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES, args['hidden_size'], NUM_OUT_FEATURES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_func = CrossEntropyLoss()

    data = data.to(DEVICE)
    pbar = tqdm(range(args['epoch']))
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
            print(f"epoch {e:03d}",
                  f"train loss {train_loss:.4f}",
                  f"train acc {train_acc:.4f}",
                  f"val acc {val_acc:.4f}",
                  )
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
