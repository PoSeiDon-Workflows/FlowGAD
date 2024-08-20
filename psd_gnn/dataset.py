""" PoSeiDon dataset wrapper as PyG.dataset.

License: MIT
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


class PSD_Dataset(InMemoryDataset):
    """ Customized dataset for PoSeiDon graphs.

    Args:
        root (str, optional):           Root of the processed path.             Defaults to "./".
        name (str, optional):           Name of the workflow type.              Defaults to "1000genome".
        use_node_attr (bool, optional): Use node attributes.                    Defaults to False.
        use_edge_attr (bool, optional): Use edge attributes.                    Defaults to False.
        force_reprocess (bool, optional): Force to reprocess, will delete the cached file on disk.
                                                                                Defaults to False.
        node_level (bool, optional):    Process as node level graphs if `True`. Defaults to False.
        binary_labels (bool, optional): Binary labels.                          Defaults to False.
        normalize (bool, optional):     Normalize the node features to [0, 1].  Defaults to True.
        anomaly_cat (list, optional):    Specify the category of anomaly, choose from ['cpu', 'hdd', 'loss', 'all'].
                                                                                Defaults to ["all"].
        anomaly_level (int, optional):  Specify the level of anomaly.           Defaults to None.
        feature_option (str, optional): Specify the feature selected, choose from ['v1', 'v2', 'v3'].
                                                                                Defaults to "v1".
        sample_size (int, optional):    Sample size for the dataset.            Defaults to None.
        transform (callable, optional): Transform function to process.          Defaults to None.
        pre_transform (callable, optional): Pre_transform function.             Defaults to None.
        pre_filter (callable, optional): Pre filter function.                   Defaults to None.
    """

    def __init__(self,
                 root="./",
                 name="1000genome",
                 use_node_attr=True,
                 use_edge_attr=False,
                 force_reprocess=False,
                 node_level=False,
                 binary_labels=False,
                 normalize=True,
                 anomaly_cat=["all"],
                 anomaly_level=None,
                 feature_option="v1",
                 sample_size=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None, **kwargs):

        self.root = root
        self.name = name.lower()
        self.use_node_attr = use_node_attr
        self.use_edge_attr = use_edge_attr
        self.force_reprocess = force_reprocess
        self.node_level = node_level
        self.binary_labels = binary_labels
        self.normalize = normalize
        self.anomaly_cat = [ac.lower() for ac in anomaly_cat]
        self.anomaly_level = anomaly_level
        self.feature_option = feature_option
        self.sample_size = sample_size

        if self.force_reprocess:
            SAVED_PATH = osp.join(osp.abspath(self.root), "processed", self.name)
            SAVED_FILE = f'{SAVED_PATH}/binary_{self.binary_labels}_node_{self.node_level}.pt'
            if osp.exists(SAVED_FILE):
                os.remove(SAVED_FILE)

        # load data if processed
        super().__init__(root, transform, pre_transform, pre_filter, **kwargs)
        self.data, self.slices, self.sizes = torch.load(self.processed_paths[0])

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
        return self.data.x.shape[1] - self.num_node_types

    @property
    def num_node_types(self):
        """ Number of node labels.

        Returns:
            int: 3. (auxiliary, compute, transfer).
        """
        return 3

    @property
    def num_edge_attributes(self):
        raise NotImplementedError

    @property
    def num_nodes_per_graph(self):
        """ Number of nodes per workflow. """
        return self.sizes['num_nodes_per_graph']

    @property
    def num_edges_per_graph(self):
        """ Number of edges per workflow. """
        return self.sizes['num_edges_per_graph']

    def process(self):
        """ Process the raw files, and save to processed files. """

        ''' process adj file '''
        if self.name in ['1000genome_new_2022', 'montage', 'predict_future_sales', 'casa-wind-full']:
            data_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "data_new")
        else:
            data_folder = osp.join(osp.dirname(osp.abspath(__file__)), "..", "data")

        ''' process adjacency file '''
        nodes, edges = parse_adj(self.name)
        n_nodes, n_edges = len(nodes), len(edges)

        # additional properties to store on disk
        sizes = {'num_nodes_per_graph': n_nodes,
                 'num_edges_per_graph': n_edges}

        self.nx_graph = nx.DiGraph(edges)

        # convert the edge_index with dim (2, E)
        edge_index = torch.tensor(edges).T

        # select features
        self.features = ['auxiliary', 'compute', 'transfer'] + \
            ['is_clustered', 'ready', 'pre_script_start',
             'pre_script_end', 'submit', 'execute_start', 'execute_end',
             'post_script_start', 'post_script_end', 'wms_delay', 'pre_script_delay',
             'queue_delay', 'runtime', 'post_script_delay', 'stage_in_delay',
             'stage_in_bytes', 'stage_out_delay', 'stage_out_bytes', 'kickstart_executables_cpu_time',
             'kickstart_status', 'kickstart_executables_exitcode']

        self.new_features = ['kickstart_online_iowait',
                             'kickstart_online_bytes_read',
                             'kickstart_online_bytes_written',
                             'kickstart_online_read_system_calls',
                             'kickstart_online_write_system_calls',
                             'kickstart_online_utime',
                             'kickstart_online_stime',
                             'kickstart_online_bytes_read_per_second',
                             'kickstart_online_bytes_written_per_second']

        self.ts_features = ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']

        self.delay_features = ["wms_delay",
                               "queue_delay",
                               "runtime",
                               "post_script_delay",
                               "stage_in_delay",
                               "stage_out_delay"]

        self.bytes_features = ["stage_in_bytes", "stage_out_bytes"]
        self.kickstart_features = ["kickstart_executables_cpu_time"]

        data_list = []
        feat_list = []

        if self.anomaly_cat == ["all"]:
            if self.name in ['1000genome_new_2022', 'montage', 'predict_future_sales', 'casa-wind-full']:
                self.y_labels = ["normal", "cpu", "hdd"]
            else:
                self.y_labels = ["normal", "cpu", "hdd", "loss"]
        else:
            if self.anomaly_level is None:
                self.y_labels = ["normal"] + self.anomaly_cat
            else:
                self.y_labels = ["normal"] + [f"{self.anomaly_cat}_{level}" for level in self.anomaly_level]

        for y_idx, y_label in enumerate(self.y_labels):
            if self.name == "1000genome_new_2022":
                raw_data_files = f"{data_folder}/{y_label}*/1000*.csv"
            else:
                raw_data_files = f"{data_folder}/{y_label}*/{self.name.replace('_', '-')}*.csv"

            assert len(glob.glob(raw_data_files)) > 0, f"Incorrect anomaly cat and level {y_label}"

            for fn in glob.glob(raw_data_files):

                # read from csv file
                df = pd.read_csv(fn, index_col=[0])
                # handle missing features
                if "kickstart_executables_cpu_time" not in df.columns:
                    continue
                # handle missing nodes (the nowind workflow)
                if df.shape[0] != len(nodes):
                    continue

                # convert `type` to dummy features
                df = pd.concat([pd.get_dummies(df.type), df], axis=1)
                df = df.drop(["type"], axis=1)
                df = df.fillna(0)

                # shift timestamp by node level
                df[self.ts_features] = df[self.ts_features].sub(df[self.ts_features].ready, axis="rows")

                # process hops
                if self.name != "predict_future_sales":
                    # TODO: fix the hops fo
                    hops = np.array([nx.shortest_path_length(self.nx_graph, 0, i) for i in range(len(nodes))])
                    df['node_hop'] = hops
                else:
                    df['node_hop'] = np.zeros(len(nodes))

                # change the index the same as `nodes`
                for i, node in enumerate(df.index.values):
                    if node.startswith("create_dir_") or node.startswith("cleanup_"):
                        new_name = node.split("-")[0]
                        df.index.values[i] = new_name

                # sort node name in json matches with node in csv.
                df = df.iloc[df.index.map(nodes).argsort()]

                # update ys from df
                if self.binary_labels:
                    if "normal" in fn:
                        y = [0] * n_nodes if self.node_level else [0]
                    else:
                        y = [1] * n_nodes if self.node_level else [1]
                else:
                    y = [y_idx] * n_nodes if self.node_level else [y_idx]
                if self.name in ['1000genome_new_2022', 'montage', 'predict_future_sales']:
                    if self.node_level:
                        # binary labels 0/1
                        y = pd.factorize(df.anomaly_type)[0]
                        # convert the `1`s to `2`s in HDD
                        if not self.binary_labels:
                            if y_label == "hdd":
                                y[np.where(y == 1)] = y_idx
                    else:
                        if not self.binary_labels:
                            y = np.array([y_idx])
                        else:
                            y = np.array([1 if y_idx > 0 else 0])
                y = torch.tensor(y)

                # extract based on selected features
                if self.feature_option == "v1":
                    selected_features = self.features + ['node_hop']
                elif self.feature_option == "v2":
                    selected_features = self.delay_features + self.bytes_features \
                        + self.kickstart_features + ['node_hop']
                elif self.feature_option == "v3":
                    selected_features = self.features + ['node_hop'] + self.new_features

                df = df[selected_features]

                # add index to node
                # df["pos"] = np.arange(df.shape[0])

                x = torch.tensor(df.to_numpy().astype(np.float32), dtype=torch.float32)
                feat_list.append(df.to_numpy())
                node_index = torch.tensor(np.arange(n_nodes), dtype=torch.long)
                data = Data(x=x,
                            node_index=node_index,
                            edge_index=edge_index,
                            y=y)
                # dump into local files
                # pk_file = fn.split("/")[-1].split(".")[0]
                # pk_path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "parsed")
                # create_dir(pk_path)
                # pickle.dump(data, osp.join(pk_path, f"{pk_file}.pkl"))
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

        if self.sample_size is not None:

            data_list = random.sample(data_list, self.sample_size)
            # save to local files
            # pk_path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "parsed")
            pk_path = osp.join("/tmp", "data", "psd", "parsed")
            create_dir(pk_path)
            file_count = 0
            for root, directories, files in os.walk(pk_path):
                file_count += len(files)
                for fn in files:
                    pre_data_list = pickle.load(open(osp.join(pk_path, fn), "rb"))
                    data_list += pre_data_list
            pickle.dump(data_list, open(osp.join(pk_path, f"iter_{file_count:04d}.pkl"), "wb"))

        # Save processed data
        if self.node_level:
            data_batch = Batch.from_data_list(data_list)
            data = Data(x=data_batch.x,
                        node_index=data_batch.node_index % n_nodes,
                        edge_index=data_batch.edge_index,
                        y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 60/20/20
            # idx = np.arange(data.num_nodes)

            # train_idx, test_idx = train_test_split(idx, train_size=0.6, random_state=0, shuffle=True)
            # val_idx, test_idx = train_test_split(test_idx, train_size=0.5, random_state=0, shuffle=True)

            # data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            # data.train_mask[train_idx] = 1

            # data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            # data.val_mask[val_idx] = 1

            # data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            # data.test_mask[test_idx] = 1
            data, slices = self.collate([data])
        else:
            data, slices = self.collate(data_list)

        # store processed data on disk
        torch.save((data, slices, sizes), self.processed_paths[0])

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
                 sample_size=None,
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
        self.anomaly_cat = [ac.lower() for ac in anomaly_cat]
        self.anomaly_level = anomaly_level
        self.sample_size = sample_size

        if self.force_reprocess:
            if osp.exists(self.processed_paths[0]):
                os.remove(self.processed_paths[0])

        self.workflows = ["1000genome",
                          "nowcast-clustering-8",
                          "nowcast-clustering-16",
                          "wind-clustering-casa",
                          "wind-noclustering-casa",
                          "1000genome_new_2022",
                          "montage"] if name == "all" else name
        # check all data are consistent and available
        for wf in self.workflows:
            _root = osp.join(osp.expanduser("~"), "tmp", "data", wf)
            print("dataset", wf)
            dataset = PSD_Dataset(root=self.root,
                                  name=wf,
                                  use_node_attr=self.use_node_attr,
                                  use_edge_attr=self.use_edge_attr,
                                  force_reprocess=self.force_reprocess,
                                  node_level=self.node_level,
                                  binary_labels=self.binary_labels,
                                  normalize=self.normalize,
                                  anomaly_cat=self.anomaly_cat,
                                  anomaly_level=self.anomaly_level,
                                  sample_size=self.sample_size)

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
        for wn in self.workflows:
            wn_path = osp.join(osp.abspath(self.root), "processed", wn)
            data = torch.load(f'{wn_path}/binary_{self.binary_labels}_node_{self.node_level}.pt')[0]
            data_list.append(data)

        if self.node_level:
            data_batch = Batch.from_data_list(data_list, exclude_keys=['node_index'])
            data = Data(x=data_batch.x,
                        # node_index=data_batch.node_index,
                        node_index=torch.concat([d.node_index for d in data_list]),
                        edge_index=data_batch.edge_index,
                        y=data_batch.y)
            data = data if self.pre_transform is None else self.pre_transform(data)

            # NOTE: split the dataset into train/val/test as 60/20/20
            # idx = np.arange(data.num_nodes)ddd
            # train_idx, test_idx = train_test_split(
            #     idx, train_size=0.6, random_state=0, shuffle=True, stratify=data.y.numpy())
            # val_idx, test_idx = train_test_split(
            #     test_idx, train_size=0.5, random_state=0, shuffle=True, stratify=data.y.numpy()[test_idx])

            # data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            # data.train_mask[train_idx] = 1

            # data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            # data.val_mask[val_idx] = 1

            # data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            # data.test_mask[test_idx] = 1

            torch.save(self.collate([data]), self.processed_paths[0])
        else:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)}) node_level {self.node_level} binary_labels {self.binary_labels}'
