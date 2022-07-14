import argparse

import torch
import torch.nn.functional as F
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.models.node_classifier import GNN
from sklearn.metrics import accuracy_score, recall_score
from torch.nn import CrossEntropyLoss, NLLLoss
from tqdm import tqdm
import numpy as np
torch.manual_seed(0)
np.random.seed(0)


def train(model, args):

    model.train()


if __name__ == "__main__":
    workflows = [
        "1000genome",
        "nowcast-clustering-8",
        "nowcast-clusteirng-16",
        "wind-clustering-casa",
        "wind-noclustering-casa"
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="1000genome", choices=workflows,
                        help="Name of the workflow.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID. -1 using CPU.")
    parser.add_argument("--epoch", type=int, default=200,
                        help="Number of epoch in training")
    args = vars(parser.parse_args())

    DEVICE = torch.device(f"cuda:{args['gpu']}") if torch.cuda.is_available() else "cpu"

    ''' binary classification '''
    dataset = PSD_Dataset("./", args['dataset'],
                          force_reprocess=True,
                          node_level=True,
                          binary_labels=True).shuffle()
    data = dataset[0]
    n_nodes = data.num_nodes

    NUM_NODE_FEATURES = dataset.num_node_features
    NUM_OUT_FEATURES = dataset.num_classes

    ''' Build GNN model '''
    model = GNN(NUM_NODE_FEATURES, 0, 64, NUM_OUT_FEATURES, 1).to(DEVICE)
    # print(model)

    lr = 1e-3
    # momentum = 0.9
    # weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # weight of classes
    # weight = F.softmax(torch.tensor(1. / data.y.bincount())).to(DEVICE)
    weight = torch.tensor(1. / data.y.bincount()).to(DEVICE)
    loss = CrossEntropyLoss(weight=weight)

    data = data.to(DEVICE)
    # add weighted loss
    results = []
    for e in range(args['epoch']):
        model.train()
        optimizer.zero_grad()
        y_hat = model(data.x, data.edge_index)

        cost = loss(y_hat[data.train_mask], data.y[data.train_mask])

        train_metric = accuracy_score(data.y[data.train_mask].detach().cpu().numpy(),
                                      y_hat[data.train_mask].argmax(dim=1).detach().cpu().numpy())
        val_metric = accuracy_score(data.y[data.val_mask].detach().cpu().numpy(),
                                    y_hat[data.val_mask].argmax(dim=1).detach().cpu().numpy())
        # test_metric = accuracy_score(data.y[data.test_mask].detach().cpu().numpy(),
        #                              y_hat[data.test_mask].argmax(dim=1).detach().cpu().numpy())
        # results.append(test_metric)
        cost.backward()
        optimizer.step()
        print(f"epoch {e:03d}",
              f"train loss {cost:.4f}",
              f"train acc {train_metric:.4f}",
              f"val acc {val_metric:.4f}",
              #   f"test acc {test_metric:.4f}"
              )

    test_metric = accuracy_score(data.y[data.test_mask].detach().cpu().numpy(),
                                 y_hat[data.test_mask].argmax(dim=1).detach().cpu().numpy())

    print(f"test acc {test_metric:.4f}")
    # print(f"best test acc {max(results):.4f}")
