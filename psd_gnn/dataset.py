""" PoSeiDon dataset wrapper as PyG.dataset.

Author: Hongwei Jin <jinh@anl.gov>
License: TBD
"""

import glob
import json
import os.path as osp
import random
import shutil

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, InMemoryDataset

from psd_gnn.utils import create_dir
from sklearn.model_selection import train_test_split


class PSD_Dataset(InMemoryDataset):

    def __init__(self,
                 root="./",
                 name="1000genome",
                 #  anomaly_type="all",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 binary_labels=False,
                 normalize=True,
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
        # self.anomaly_type = anomaly_type.lower()
        # assert self.anomaly_type in ["all", "cpu", "hdd", "loss"]
        # self.attr_options = attr_options.lower()
        # assert self.attr_options in ["s1", "s2", "s3"]
        # force to reprocess again
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.binary_labels = binary_labels
        self.normalize = normalize
        if self.force_reprocess:
            SAVED_PATH = osp.join(osp.abspath(root), "processed", self.name)
            if osp.exists(SAVED_PATH):
                shutil.rmtree(SAVED_PATH)
        # load data if processed
        super().__init__(root, transform, pre_transform, pre_filter)
        # if self.force_reprocess:
        #     os.remove(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.

        Returns:
            list: List of file names.
        """
        SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
        create_dir(SAVED_PATH)
        return [f'{SAVED_PATH}/data_binary_{self.binary_labels}_node_{self.node_level}.pt']

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
        adj_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "adjacency_list_dags")
        data_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "data")
        if self.name == "all":
            # TODO: process the entire graphs
            pass
        else:
            adj_file = osp.join(adj_folder, f"{self.name.replace('-', '_')}.json")
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

        features = ['auxiliary', 'compute', 'transfer'] + \
            ['is_clustered', 'ready', 'pre_script_start',
             'pre_script_end', 'submit', 'execute_start', 'execute_end',
             'post_script_start', 'post_script_end', 'wms_delay', 'pre_script_delay',
             'queue_delay', 'runtime', 'post_script_delay', 'stage_in_delay',
             'stage_in_bytes', 'stage_out_delay', 'stage_out_bytes', 'kickstart_executables_cpu_time',
             'kickstart_status', 'kickstart_executables_exitcode']
        data_list = []
        feat_list = []
        n = len(nodes)
        for filename in glob.glob(f"{data_folder}/*/{self.name.replace('_', ' - ')}*.csv"):
            # process labels according to classes
            if self.binary_labels:
                if "normal" in filename:
                    y = torch.tensor([0]) if not self.node_level else torch.tensor([0] * n)
                else:
                    y = torch.tensor([1]) if not self.node_level else torch.tensor([1] * n)
            else:
                if "normal" in filename:
                    y = torch.tensor([0]) if not self.node_level else torch.tensor([0] * n)
                elif "cpu" in filename:
                    y = torch.tensor([1]) if not self.node_level else torch.tensor([1] * n)
                elif "hdd" in filename:
                    y = torch.tensor([2]) if not self.node_level else torch.tensor([2] * n)
                elif "loss" in filename:
                    y = torch.tensor([3]) if not self.node_level else torch.tensor([3] * n)

            df = pd.read_csv(filename, index_col=[0])

            # handle missing features
            if "kickstart_executables_cpu_time" not in df.columns:
                continue
            # handle the nowind workflow
            if df.shape[0] != len(nodes):
                continue

            # convert type to dummy features
            df = pd.concat([pd.get_dummies(df.type), df], axis=1)
            df = df.drop(["type"], axis=1)
            df = df[features]
            df = df.fillna(0)

            # shift timestamps
            ts_anchor = df['ready'].min()
            for attr in ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']:
                df[attr] -= ts_anchor

            # change the index same with `nodes`
            for i, node in enumerate(df.index.values):
                if node.startswith("create_dir_") or node.startswith("cleanup_"):
                    new_name = node.split("-")[0]
                    df.index.values[i] = new_name

            x = torch.tensor(df.to_numpy(), dtype=torch.float32)
            feat_list.append(df.to_numpy())
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # normalize across jobs
        # backend: numpy
        if self.normalize:
            all_feat = np.array(feat_list)
            # v_min = all_feat.min(axis=1, keepdims=True)
            # v_max = all_feat.max(axis=1, keepdims=True)
            v_min = np.concatenate(all_feat).min(axis=0)
            v_max = np.concatenate(all_feat).max(axis=0)
            norm_feat = (all_feat - v_min) / (v_max - v_min)
            np.nan_to_num(norm_feat, 0)
            for i, x in enumerate(norm_feat):
                data_list[i].x = torch.tensor(x, dtype=torch.float32)

        # backend: pytorch
        # if self.normalize:
        #     all_feat = torch.stack(feat_list)
        #     v_min = torch.min(all_feat, dim=1, keepdim=True)[0]
        #     v_max = torch.max(all_feat, dim=1, keepdim=True)[0]
        #     norm_feat = (all_feat - v_min) / (v_max - v_min)
        #     torch.nan_to_num(norm_feat, 0, 1, 0)
        #     for i, x in enumerate(norm_feat):
        #         data_list[i].x = torch.tensor(x, dtype=torch.float32)

        random.shuffle(data_list)

        # Save processed data
        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)
            # TODO update with train_mask, val_mask, test_mask
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'
