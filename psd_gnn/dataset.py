""" PoSeiDon dataset wrapper as PyG.dataset.

License: TBD
"""

import glob
import os
import os.path as osp
import pickle

import random
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data, InMemoryDataset

from psd_gnn.utils import create_dir, parse_adj
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


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
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.binary_labels = binary_labels
        self.normalize = normalize
        self.anomaly_cat = anomaly_cat.lower()
        self.anomaly_level = anomaly_level

        if self.force_reprocess:
            SAVED_PATH = osp.join(osp.abspath(root), "processed", self.name)
            SAVED_FILE = f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt'
            SAVED_FILE2 = f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pkl'
            if osp.exists(SAVED_FILE):
                os.remove(SAVED_FILE)
                os.remove(SAVED_FILE2)

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
        return [f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt',
                f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pkl']

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
            norm_feat = (all_feat - v_min) / (v_max - v_min) + 0
            np.nan_to_num(norm_feat, False)
            for i, x in enumerate(norm_feat):
                data_list[i].x = torch.tensor(x, dtype=torch.float32)

        pickle.dump(data_list, open(self.processed_file_names[1], "wb"))
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
            # TODO: add data_loader to the fixed split of dataset

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'


class PSD_Dataset_v2(InMemoryDataset):
    """ Old normalizing process """

    def __init__(self,
                 root="./",
                 name="1000genome",
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
        return [f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt']

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

            # NOTE: old normalization trick
            graph = {"y": y,
                     "edge_index": edge_index,
                     "x": []}
            df = pd.read_csv(filename, index_col=[0])

            df = df.replace("", -1, regex=True)

            # handle missing features
            if "kickstart_executables_cpu_time" not in df.columns:
                continue
            # handle the nowind workflow
            if df.shape[0] != len(nodes):
                continue

            # change the index same with `nodes`
            for l in nodes:
                if l.startswith("create_dir_") or l.startswith("cleanup_"):
                    new_l = l.split("-")[0]
                else:
                    new_l = l

                job_features = df[df.index.str.startswith(new_l)][['type', 'ready',
                                                                   'submit', 'execute_start', 'execute_end', 'post_script_start',
                                                                   'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
                                                                   'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']].values.tolist()[0]

                if len(df[df.index.str.startswith(new_l)]) < 1:
                    continue
                if job_features[0] == 'auxiliary':
                    job_features[0] = 0
                if job_features[0] == 'compute':
                    job_features[0] = 1
                if job_features[0] == 'transfer':
                    job_features[0] = 2
                job_features = [-1 if x != x else x for x in job_features]
                graph['x'].insert(nodes[l], job_features)

            t_list = []
            for i in range(len(graph['x'])):
                t_list.append(graph['x'][i][1])
            minim = min(t_list)

            for i in range(len(graph['x'])):
                lim = graph['x'][i][1:7]
                lim = [v - minim for v in lim]
                graph['x'][i][1:7] = lim

            gx = torch.tensor(np.array(graph['x']), dtype=torch.float32)
            v_min, v_max = gx.min(), gx.max()
            new_min, new_max = -1, 1
            gx = (gx - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
            data = Data(x=gx, edge_index=edge_index, y=y)
            data_list.append(data)

        # Save processed data
        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x, edge_index=data_batch.edge_index, y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 20/20/60
            idx = np.arange(data.num_nodes)
            train_idx, test_idx = train_test_split(
                idx, test_size=0.6, random_state=0, shuffle=True, stratify=data.y.numpy())
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.5, random_state=0, shuffle=True, stratify=data.y.numpy()[train_idx])

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
            # TODO: add data_loader to the fixed split of dataset

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'


class Merge_PSD_Dataset(InMemoryDataset):
    def __init__(self, root="./",
                 node_level=True,
                 binary_labels=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None) -> None:
        self.name = "all"
        self.root = root
        self.node_level = node_level
        self.binary_labels = binary_labels
        workflows = ["1000genome",
                     "nowcast-clustering-8",
                     "nowcast-clustering-16",
                     "wind-clustering-casa",
                     "wind-noclustering-casa"]
        # check all data are available
        for w in workflows:
            dataset = PSD_Dataset(name=w, node_level=self.node_level, binary_labels=self.binary_labels)
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
        return [f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pt',
                f'{SAVED_PATH}/{self.__class__.__name__}_binary_{self.binary_labels}_node_{self.node_level}.pkl']

    def process(self):
        """ process """
        data_list = []
        for wn in ["1000genome",
                   "nowcast-clustering-8",
                   "nowcast-clustering-16",
                   "wind-clustering-casa",
                   "wind-noclustering-casa"]:
            wn_path = osp.join(osp.abspath(self.root), "processed", wn)
            subdata_list = pickle.load(
                open(
                    f'{wn_path}/PSD_Dataset_binary_{self.binary_labels}_node_{self.node_level}.pkl',
                    'rb'))
            data_list += subdata_list
        pickle.dump(data_list, open(self.processed_file_names[1], "wb"))

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