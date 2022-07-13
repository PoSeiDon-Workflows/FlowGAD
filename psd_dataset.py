import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from psd_gnn.base_model import GNN
from psd_gnn.dataset import PSD_Dataset

if __name__ == "__main__":
    # dataset stats
    torch.manual_seed(0)
    np.random.seed(0)

    for ds in ["1000genome",
               "nowcast-clustering-8",
               "nowcast-clustering-16",
               "wind-clustering-casa",
               "wind-noclustering-casa"]:

        dataset = PSD_Dataset(root='./', name=ds)
        print(f"dataset                 {ds} \n",
              f"# of graphs             {len(dataset)} \n",
              f"# of graph labels       {dataset.num_classes} \n",
              f"# of node labels        {dataset.num_node_labels} \n",
              f"# of node features      {dataset.num_node_features} \n",
              f"# of nodes per graph    {dataset[0].num_nodes} \n",
              f"# of edges per graph    {dataset[0].num_edges} \n",
              "##" * 20 + "\n"
              )
    # exit()
    # taking 1000genome as demo
    dataset = PSD_Dataset(
        root='./',
        name="wind-noclustering-casa",
        transform=T.NormalizeFeatures()).shuffle()
    print(dataset)
    n_graphs = len(dataset)

    train_idx, test_idx = train_test_split(np.arange(n_graphs), test_size=0.2)
    train_idx, val_idx = train_test_split(train_idx)
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    # train_dataset = dataset[len(dataset) // 10:]
    # test_dataset = dataset[:len(dataset) // 10]
    train_loader = DataLoader(train_dataset, 1, shuffle=True)
    val_loader = DataLoader(val_dataset, 1)
    test_loader = DataLoader(test_dataset, 1)

    # TODO: DeepHyper to take NAS

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    model = GNN(dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def train(epoch):
        model.train()

        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / len(train_dataset)

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    for epoch in range(1, 201):
        loss = train(epoch)
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
              f'Val Acc: {val_acc:.5f}')

    print(f"Test acc {test(test_loader):.5f}")
