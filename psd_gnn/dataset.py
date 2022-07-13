# -*- coding: utf-8 -*-
""" PoSeiDon dataset wrapper for PyG.

Author: Hongwei Jin <jinh@anl.gov>
License: TBD
"""

import glob
import json
import os.path as osp

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from psd_gnn.base_model import GNN
from psd_gnn.utils import create_dir


class PSD_Dataset(InMemoryDataset):

    def __init__(self,
                 root="./",
                 name="1000genome",
                 anomaly_type="all",
                 attr_options="s1",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 binary_labels=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        """ Customized dataset for PoSeiDon graphs.

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
        self.name = name.lower()
        self.anomaly_type = anomaly_type.lower()
        assert self.anomaly_type in ["all", "cpu", "hdd", "loss"]
        self.attr_options = attr_options.lower()
        assert self.attr_options in ["s1", "s2", "s3"]
        # TODO: force to reprocess again
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.binary_labels = binary_labels
        super().__init__(root, transform, pre_transform, pre_filter)
        # load data if processed
        if not self.force_reprocess:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        """ Processed file name(s).

        Returns:
            list: List of file names.
        """
        SAVED_PATH = osp.join(osp.abspath(self.root), "processed", f"{self.name}")
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/data_{self.name}.pt']

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
        ''' process adj file '''
        # TODO: process the entire graphs
        if self.name == "all":
            pass
        else:
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

        # binary labels / multi lables
        if self.binary_labels:
            classes = {"normal": 0, "cpu": 1, "hdd": 1, "loss": 1}
        else:
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
            # TODO: process labels according to classes
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
