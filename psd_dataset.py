# -*- coding: utf-8 -*-
""" PoSeiDon dataset wrapper for PyG.

Author: Hongwei Jin <jinh@anl.gov>
License: TBD
"""

import glob
import json
import os
import os.path as osp

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class PSD_Dataset(InMemoryDataset):

    def __init__(self,
                 root="./",
                 name="1000genome",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """ Init the PoSeiDon dataset.

        Args:
            root (str, optional): Root of the processed path. Defaults to "./".
            name (str, optional): Name of the workflow type. Defaults to "1000genome".
            use_node_attr (bool, optional): Use node attributes. Defaults to False.
            use_edge_attr (bool, optional): Use edge attributes. Defaults to False.
            force_reprocess (bool, optional): Force to reprocess. Defaults to False.
            node_level (bool, optional): Process as node level graphs if `True`. Defaults to False.
            transform (callable, optional): Transform function to process. Defaults to None.
            pre_transform (callable, optional): Pre_transform function. Defaults to None.
            pre_filter (callable, optional): Pre filter function. Defaults to None.
        """
        self.name = name
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        super().__init__(root, transform, pre_transform, pre_filter)
        # load data if processed
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        """ Processed file name(s).

        Returns:
            list: List of file names.
        """
        SAVED_PATH = osp.join(osp.abspath(self.root), "processed", f"{self.name.lower()}")
        if not osp.exists(SAVED_PATH):
            os.makedirs(SAVED_PATH)
        return [f'{SAVED_PATH}/data_{self.name.lower()}.pt']

    @property
    def num_node_attributes(self):
        """ Number of node features. """
        if self.data.x is None:
            return 0
        return self.data.x.shape[1] - self.num_node_labels

    @property
    def num_node_labels(self):
        """ Number of node labels.

        Returns:
            int: 3. (auxiliary, compute, transfer).
        """
        return 3

    @property
    def num_edge_attributes(self):
        raise NotImplementedError

    def process(self):
        """ Process the raw files, and save to processed files. """
        # process adj file
        adj_file = f"./adjacency_list_dags/{self.name.replace('-', '_')}.json"
        d = json.load(open(adj_file))

        # build dict of nodes
        nodes = {k: i for i, k in enumerate(d.keys())}

        # clean up with no timestamps in nodes
        nodes = {}
        for i, k in enumerate(d.keys()):
            if k.startswith("create_dir_") or k.startswith("cleanup_"):
                k = k.split("-")[0]
                nodes[k] = i
            else:
                nodes[k] = i

        edges = []
        for u in d:
            for v in d[u]:
                if u.startswith("create_dir_") or u.startswith("cleanup_"):
                    u = u.split("-")[0]
                if v.startswith("create_dir_") or v.startswith("cleanup_"):
                    v = v.split("-")[0]
                edges.append((nodes[u], nodes[v]))

        # convert the edge_index with dim (2, E)
        edge_index = torch.tensor(edges).T

        classes = {"normal": 0, "cpu": 1, "hdd": 2, "loss": 3}

        # REVIEW: verify the features.
        features = ['type',
                    # 'ready',
                    # 'submit',
                    'wms_delay',
                    'pre_script_delay',
                    'queue_delay',
                    'runtime',
                    'post_script_delay',
                    'stage_in_delay',
                    'stage_out_delay']

        # features = ['auxiliary', 'compute', 'transfer',
        #             'pre_script_delay',
        #             'post_script_delay',
        #             'wms_delay',
        #             'queue_delay',
        #             'runtime',
        #             'stage_in_delay',
        #             'stage_out_delay']

        data_list = []
        n = len(nodes)
        for filename in glob.glob(f"./data/*/{self.name.replace('_', ' - ')}*.csv"):
            if "normal" in filename:
                y = torch.tensor([0]) if not self.node_level else torch.tensor([0] * n)
            elif "cpu" in filename:
                y = torch.tensor([1]) if not self.node_level else torch.tensor([0] * n)
            elif "hdd" in filename:
                y = torch.tensor([2]) if not self.node_level else torch.tensor([0] * n)
            elif "loss" in filename:
                y = torch.tensor([3]) if not self.node_level else torch.tensor([0] * n)

            df = pd.read_csv(filename, index_col=[0])

            df = df[features]
            # change the index same with `nodes`
            for i, n in enumerate(df.index.values):
                if n.startswith("create_dir_") or n.startswith("cleanup_"):
                    new_name = n.split("-")[0]
                    df.index.values[i] = new_name

            # sort with nodes index
            df = df.iloc[df.index.map(nodes).argsort()]
            # convert type into hot vectors
            df = pd.concat([pd.get_dummies(df.type), df], axis=1)
            # remove `type`
            df = df.drop(["type"], axis=1)
            # fill NaN with 0
            df = df.fillna(0)

            # REVIEW: convert ts to gaps
            # df['ready'] = df['ready'] - df['ready'].min()
            # df['submit'] = df['submit'] - df['submit'].min()

            x = torch.tensor(df.to_numpy(), dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)})'


if __name__ == "__main__":
    # dataset stats
    import numpy as np
    from sklearn.model_selection import train_test_split
    torch.manual_seed(0)
    np.random.seed(0)

    # for ds in ["1000genome",
    #            "nowcast-clustering-8",
    #            "nowcast-clustering-16",
    #            "wind-clustering-casa",
    #            "wind-noclustering-casa"]:
    #     dataset = PSD_Dataset(root='./', name=ds)
    #     print(f"dataset                 {ds} \n",
    #           f"# of graphs             {len(dataset)} \n",
    #           f"# of graph labels       {dataset.num_classes} \n",
    #           f"# of node labels        {dataset.num_node_labels} \n",
    #           f"# of node features      {dataset.num_node_features} \n",
    #           f"# of nodes per graph    {dataset[0].num_nodes} \n",
    #           f"# of edges per graph    {dataset[0].num_edges} \n",
    #           "##" * 20 + "\n"
    #           )

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
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = GraphConv(dataset.num_features, 128)
            self.pool1 = TopKPooling(128, ratio=0.8)
            self.conv2 = GraphConv(128, 128)
            self.pool2 = TopKPooling(128, ratio=0.8)
            self.conv3 = GraphConv(128, 128)
            self.pool3 = TopKPooling(128, ratio=0.8)

            self.lin1 = torch.nn.Linear(256, 128)
            self.lin2 = torch.nn.Linear(128, 64)
            self.lin3 = torch.nn.Linear(64, dataset.num_classes)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = x1 + x2 + x3

            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.log_softmax(self.lin3(x), dim=-1)

            return x

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
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
