""" PoSeiDon dataset wrapper as PyG.dataset.

License: TBD
"""

import glob
import os
import os.path as osp
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data, InMemoryDataset

from psd_gnn.utils import create_dir, parse_adj


class PSD_Dataset(InMemoryDataset):
    """ New normalizing process """

    def __init__(self,
                 root="./",
                 name="1000genome",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 binary_labels=False,
                 normalize=True,
                 anomaly_cat="all",
                 anomaly_level=None,
                 anomaly_num=None,
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
            binary_labels (bool, optional): Binary labels. Defaults to False.
            normalize (bool, optional): Normalize the node features to [0, 1]. Defaults to True.
            anomaly_cat (str, optional): Specify the category of anomaly, choose from ['cpu', 'hdd', 'loss', 'all'].
                                         Defaults to "all".
            anomaly_level (_type_, optional): Specify the level of anomaly. Defaults to None.
            anomaly_num (_type_, optional): Specify the number of anomaly. Defaults to None. Warning: this will be removed.
            transform (callable, optional): Transform function to process. Defaults to None.
            pre_transform (callable, optional): Pre_transform function. Defaults to None.
            pre_filter (callable, optional): Pre filter function. Defaults to None.
        """
        self.root = root
        self.name = name.lower()
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.binary_labels = binary_labels
        self.normalize = normalize
        self.anomaly_cat = anomaly_cat.lower()
        self.anomaly_level = anomaly_level
        self.anomaly_num = anomaly_num

        if self.force_reprocess:
            SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
            SAVED_FILE = f'{SAVED_PATH}/binary_{self.binary_labels}_node_{self.node_level}.pt'
            if osp.exists(SAVED_FILE):
                os.remove(SAVED_FILE)

        # load data if processed
        super().__init__(root, transform, pre_transform, pre_filter)
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
        return [f'{SAVED_PATH}/binary_{self.binary_labels}_node_{self.node_level}.pt']

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
        data_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "data")

        nodes, edges = parse_adj(self.name)

        self.num_nodes_per_graph = len(nodes)
        self.num_edges_per_graph = len(edges)
        self.nx_graph = nx.DiGraph(edges)
        # convert the edge_index with dim (2, E)
        edge_index = torch.tensor(edges).T

        self.features = ['auxiliary', 'compute', 'transfer'] + \
            ['is_clustered', 'ready', 'pre_script_start',
             'pre_script_end', 'submit', 'execute_start', 'execute_end',
             'post_script_start', 'post_script_end', 'wms_delay', 'pre_script_delay',
             'queue_delay', 'runtime', 'post_script_delay', 'stage_in_delay',
             'stage_in_bytes', 'stage_out_delay', 'stage_out_bytes', 'kickstart_executables_cpu_time',
             'kickstart_status', 'kickstart_executables_exitcode']
        data_list = []
        feat_list = []

        if self.anomaly_cat == "all":
            self.y_labels = ["normal", "cpu", "hdd", "loss"]
        else:
            if self.anomaly_level is None:
                self.y_labels = ["normal", self.anomaly_cat]
            else:
                self.y_labels = ["normal"] + [f"{self.anomaly_cat}_{level}" for level in self.anomaly_level]

        for y_idx, y_label in enumerate(self.y_labels):
            raw_data_files = f"{data_folder}/{y_label}*/{self.name.replace('_', ' - ')}*.csv"
            assert len(glob.glob(raw_data_files)) > 0, f"Incorrect anomaly cat and level {y_label}"

            for fn in glob.glob(raw_data_files):
                if self.binary_labels:
                    if "normal" in fn:
                        y = [0] * self.num_nodes_per_graph if self.node_level else [0]
                    else:
                        y = [1] * self.num_nodes_per_graph if self.node_level else [1]
                else:
                    y = [y_idx] * self.num_nodes_per_graph if self.node_level else [y_idx]
                y = torch.tensor(y)

                df = pd.read_csv(fn, index_col=[0])
                # handle missing features
                if "kickstart_executables_cpu_time" not in df.columns:
                    continue
                # handle the nowind workflow
                if df.shape[0] != len(nodes):
                    continue

                # convert type to dummy features
                df = pd.concat([pd.get_dummies(df.type), df], axis=1)
                df = df.drop(["type"], axis=1)
                df = df[self.features]
                df = df.fillna(0)

                # shift timestamps
                ts_anchor = df['ready'].min()
                for attr in ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']:
                    df[attr] -= ts_anchor

                # change the index the same as `nodes`
                for i, node in enumerate(df.index.values):
                    if node.startswith("create_dir_") or node.startswith("cleanup_"):
                        new_name = node.split("-")[0]
                        df.index.values[i] = new_name

                # sort node name in json matches with node in csv.
                df = df.iloc[df.index.map(nodes).argsort()]
                hops = np.array([nx.shortest_path_length(self.nx_graph, 0, i) for i in range(len(nodes))])
                # REVIEW: add node_id to feature -> nodes with different ids have different patterns
                # df['node_id'] = np.arange(self.num_nodes_per_graph)
                # df['node_hop'] = hops / hops.max()

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
            v_min = np.concatenate(all_feat).min(axis=(0))
            v_max = np.concatenate(all_feat).max(axis=(0))
            norm_feat = (all_feat - v_min) / (v_max - v_min)
            np.nan_to_num(norm_feat, False)
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

        # Save processed data
        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 60/20/20
            idx = np.arange(data.num_nodes)

            # train_idx, test_idx = train_test_split(
            #     idx, train_size=0.6, random_state=0, shuffle=True, stratify=data.y.numpy())
            # val_idx, test_idx = train_test_split(
            #     test_idx, train_size=0.5, random_state=0, shuffle=True, stratify=data.y.numpy()[test_idx])

            train_idx, test_idx = train_test_split(idx, train_size=0.6, random_state=0, shuffle=True)
            val_idx, test_idx = train_test_split(test_idx, train_size=0.5, random_state=0, shuffle=True)

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = 1

            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = 1

            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = 1
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'


class Merge_PSD_Dataset(InMemoryDataset):
    def __init__(self, root="./",
                 name="all",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 binary_labels=False,
                 normalize=True,
                 anomaly_cat="all",
                 anomaly_level=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None) -> None:
        self.root = root
        self.name = "all"
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.binary_labels = binary_labels
        self.normalize = normalize
        self.anomaly_cat = anomaly_cat.lower()
        self.anomaly_level = anomaly_level

        workflows = ["1000genome",
                     "nowcast-clustering-8",
                     "nowcast-clustering-16",
                     "wind-clustering-casa",
                     "wind-noclustering-casa"]
        # check all data are consistent and available
        for wf in workflows:
            dataset = PSD_Dataset(root=self.root,
                                  name=wf,
                                  use_node_attr=self.use_node_attr,
                                  use_edge_attr=self.use_edge_attr,
                                  force_reprocess=self.force_reprocess,
                                  node_level=self.node_level,
                                  binary_labels=self.binary_labels,
                                  normalize=self.normalize,
                                  anomaly_cat=self.anomaly_cat,
                                  anomaly_level=self.anomaly_level)

        super().__init__(root, transform, pre_transform, pre_filter)
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
        return [f'{SAVED_PATH}/binary_{self.binary_labels}_node_{self.node_level}.pt']

    def process(self):
        """ process """
        data_list = []
        for wn in ["1000genome",
                   "nowcast-clustering-8",
                   "nowcast-clustering-16",
                   "wind-clustering-casa",
                   "wind-noclustering-casa"]:
            wn_path = osp.join(osp.abspath(self.root), "processed", wn)
            data = torch.load(f'{wn_path}/binary_{self.binary_labels}_node_{self.node_level}.pt')[0]
            data_list.append(data)

        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 60/20/20
            idx = np.arange(data.num_nodes)
            train_idx, test_idx = train_test_split(
                idx, train_size=0.6, random_state=0, shuffle=True, stratify=data.y.numpy())
            val_idx, test_idx = train_test_split(
                test_idx, train_size=0.5, random_state=0, shuffle=True, stratify=data.y.numpy()[test_idx])

            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[train_idx] = 1

            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[val_idx] = 1

            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[test_idx] = 1
            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'
