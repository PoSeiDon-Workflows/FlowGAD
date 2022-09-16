""" Demo of graph-level generative model. """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.models.psd_gan_v1 import Dis, Gen
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import pickle

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = PSD_Dataset("./", node_level=False, binary_labels=True)

    # process the train / test split
    Xs, ys = [], []
    for data in dataset:
        Xs.append(data.x.numpy())
        ys.append(data.y.item())
    Xs, ys = np.array(Xs), np.array(ys)
    idx = np.arange(len(dataset))

    ys_0 = idx[ys == 0]
    ys_1 = idx[ys == 1]

    train_0, test_0 = train_test_split(ys_0, train_size=100, random_state=0)
    train_1, test_1 = train_test_split(ys_1, train_size=100, random_state=0)

    # NOTE: extract the REAL anomaly distribution
    Xs_train_1 = torch.tensor(Xs[train_1], dtype=torch.float)

    train_idx = np.concatenate([train_0, train_1])
    test_idx = np.concatenate([test_0, test_1])

    train_ys, test_ys = ys[train_idx], ys[test_idx]
    train_loader = DataLoader(dataset[train_idx], batch_size=1)
    test_loader = DataLoader(dataset[test_idx], batch_size=1)

    # init the Gen/Dis networks
    net_G = Gen(in_dim=24, hid_dim=64, out_dim=24).to(device)
    net_D = Dis(in_dim=24, hid_dim=64, out_dim=2).to(device)

    optim_G = torch.optim.Adam(net_G.parameters(), lr=1e-3)
    optim_D = torch.optim.Adam(net_D.parameters(), lr=1e-3)

    losses_G = []
    losses_D = []

    feat_list = []

    criterion = nn.BCELoss()
    net_D.train()
    net_G.train()

    # prepare the label
    label_0 = torch.zeros((dataset.num_node_features, 2), device=device, dtype=torch.float)
    label_0[:, 0] = 1.
    label_1 = torch.zeros((dataset.num_node_features, 2), device=device, dtype=torch.float)
    label_1[:, 1] = 1.

        loss_g, loss_d = 0, 0

        for i, data in enumerate(train_loader):
            data = data.to(device)
            label_real = torch.zeros((data.x.shape[0], 2), device=device, dtype=torch.float)
            label_real[:, 0] = 1.

            label_fake = torch.zeros((data.x.shape[0], 2), device=device, dtype=torch.float)
            label_fake[:, 1] = 1.

            # 1. update net_D
            # pass real data
            optim_D.zero_grad()
            output_real = net_D(data.x, data.edge_index)
            logits_real = F.softmax(output_real, dim=1)
            label_real = torch.zeros((data.x.shape[0], 2), device=device, dtype=torch.float)
            label_real[:, 0] = 1.
            err_D_real = criterion(logits_real, label_real)
            err_D_real.backward()

            # pass fake data
            noise = torch.randn(data.x.shape, device=device) + data.x
            # noise = torch.clip(noise, 0)
            fake_x = net_G(noise)
            label_fake = torch.zeros((data.x.shape[0], 2), device=device, dtype=torch.float)
            label_fake[:, 1] = 1.
            # fake_x = torch.clip(fake_x, min=0, max=1)
            output_fake = net_D(fake_x, data.edge_index)
            logits_fake = F.softmax(output_fake, dim=1)
            err_D_fake = criterion(logits_fake, label_fake)
            err_D_fake.backward(retain_graph=True)

            err_D = (err_D_real + err_D_fake) / 2
            optim_D.step()

            # 2. update net_G
            optim_G.zero_grad()
            output_fake = net_D(fake_x, data.edge_index)
            logits_fake = F.softmax(output_fake, dim=1)
            label_fake = torch.zeros((data.x.shape[0], 2), device=device, dtype=torch.float)
            label_fake[:, 1] = 1.
            err_G = criterion(logits_fake, label_real)
            err_G.backward()
            optim_G.step()

            # aggregate loss
            loss_d += err_D.item() * data.num_graphs
            loss_g += err_G.item() * data.num_graphs

        losses_G.append(loss_g)
        losses_D.append(loss_d)
        # feat_list.append(fake_x.detach().cpu().numpy())
        print(f"epoch {e:03d} loss_D {loss_d:.4f} loss_G {loss_g:.4f}")

    for data in train_loader:
        data = data.to(device)
        fake_xs = net_G(data.x)
        fake_xs = torch.split(fake_xs, data.num_graphs)
        for fake_x in fake_xs:
            feat_list.append(fake_x.detach().cpu().numpy())

    # pickle.dump({'losses_G': losses_G, 'losses_D': losses_D, 'feat_list': feat_list}, open("gan_res_v1.pkl", "wb"))

    net_D.eval()
    count = 0
    y_pred = []
    y_true = []
    for data in test_loader:
        data = data.to(device)
        output_real = net_D(data.x, data.edge_index)
        logits_real = F.softmax(output_real, dim=1)
        # label_real = torch.zeros((data.x.shape[0], 2), device=device, dtype=torch.float)
        # label_real[:, 1] = 1.
        # REVIEW: majority vote on predictions
        # graph-level
        pred_label = logits_real.argmax(dim=1).bincount().argmax().detach().cpu().item()
        y_pred.append(pred_label)
        y_true.append(data.y.detach().cpu().item())

    print(f"acc {accuracy_score(y_true, y_pred):.4f}",
          f"f1 {f1_score(y_true, y_pred, average='weighted'):.4f}",
          f"recall {recall_score(y_true, y_pred, average='weighted'):.4f}")

    print(confusion_matrix(y_true, y_pred))
