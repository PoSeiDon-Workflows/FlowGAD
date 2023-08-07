# imports
import os.path as osp
import warnings
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from pygod.metrics import (eval_average_precision, eval_precision_at_k,
                           eval_recall_at_k, eval_roc_auc)
from scipy.special import erf
from scipy.stats import binom
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj


import glob
import os
import os.path as osp

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data, InMemoryDataset

from psd_gnn.utils import create_dir, parse_adj

from psd_gnn.dataset import PSD_Dataset


def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape)
    g = -torch.log(-torch.log(unif + eps))
    return g 

def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar
        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = sample_gumbel(logits.shape)
    h = (g + logits)/temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y


""" utility functions and classes """
def validate_device(gpu):
    """ Validate GPU device. """
    gpu_id = int(gpu)
    if gpu_id >= 0 and torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    return device

class MinMaxNormalizeFeatures(BaseTransform):
    r"""Min-max normalizes the attributes given in :obj:`attrs` to scale between 0 and 1.
    (functional name: :obj:`minmax_normalize_features`).
    Args:
        attrs (List[str], optional): The names of attributes to normalize. Defaults to ["x"].
    """

    def __init__(self, attrs: List[str] = ["x"],
                 min: int = 0,
                 max: int = 1) -> None:
        self.attrs = attrs
        self.min = min
        self.max = max

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                # add a small eps for nan values
                value = value.sub(value.min(dim=0)[0]).div(value.max(dim=0)[0].sub(
                    value.min(dim=0)[0] + 1e-10))
                value = value * (self.max - self.min) + self.min
                store[key] = value
        return data
    

""" SSL and SSL_base classes """
class SSL(nn.Module):
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.5,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=None,
                 eta=.5,
                 contamination=0.05,
                 lr=5e-3,
                 epoch=200,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 margin=.5,
                 r=.2,
                 m=50,
                 k=50,
                 f=10,
                 K=10,
                 N=2,
                temperature=1,
                verbose=False):

        super(SSL, self).__init__()
        assert 0. < contamination <= 0.5,\
              ValueError(f"contamination must be in (0, 0.5], got: {contamination:.2f}")


        self.contamination = contamination
        self.decision_scores_ = None

        
        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.eta = eta


        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh
        self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)

        # other param
        self.verbose = verbose
        self.r = r
        # self.m = m
        self.k = k
        self.f = f
        self.model = None
        self.N=N
        self.K=K
        self.temperature=temperature


    def fit(self, data, y_true=None, temperature = 1, training=True):
        """
        Fit detector with input data.

        Parameters
        ----------
        data : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        
        data.node_idx = torch.arange(data.x.shape[0])


        # automated balancing by std
        if self.alpha is None:
            adj = to_dense_adj(data.edge_index)[0]
            self.alpha = torch.std(adj).detach() / \
                (torch.std(data.x).detach() + torch.std(adj).detach())
            adj = None


        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]


        loader = NeighborLoader(data,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)


        self.model = SSL_Base(in_dim=data.x.shape[1],
                                hid_dim=self.hid_dim,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                act=self.act).to(self.device)


        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)


        self.model.train()
        temp_min= 0.001
        ANNEAL_RATE = 0.0002


        # decision scores for each node
        decision_scores = np.zeros(data.x.shape[0])

        for epoch in range(self.epoch):

            epoch_loss = 0
            for sampled_data in loader:


                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, edge_index = self.process_graph(sampled_data)


                # generate augmented graph
                x_aug, label_aug = self._data_augmentation(x)
                h_aug = self.model.embed(x_aug, edge_index)
                h = self.model.embed(x, edge_index)
                h = F.log_softmax(h.view(-1, self.N, self.K), dim=-1)
                h_aug = F.log_softmax(h.view(-1, self.N, self.K), dim=-1)
                

                # Sampling
                h = sample_gumbel_softmax(h, self.temperature).view(-1, self.N*self.K)
                h_aug = sample_gumbel_softmax(h_aug, self.temperature).view(-1, self.N*self.K)


                # margin loss
                margin_loss = self.margin_loss_func(h, h_aug, h) * label_aug
                margin_loss = torch.mean(margin_loss)


                # reconstruction loss
                x_ = self.model.reconstruct(h, edge_index)
                score = self.loss_func(x[:batch_size], x_[:batch_size])


                # NEW
                # score = self.loss_func(x, x_)
                reconstruct_loss = torch.mean(score)


                # total loss
                loss = self.eta * reconstruct_loss + (1 - self.eta) * margin_loss
                decision_scores[node_idx[:batch_size]] = score.detach().cpu().numpy()
                epoch_loss += loss.item() * batch_size


                # NEW
                # decision_scores[node_idx] = score.detach().cpu().numpy()
                # epoch_loss += loss.item() * x.shape[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if epoch % 10==0:
                self.temperature = np.maximum(self.temperature * np.exp(-ANNEAL_RATE * epoch), temp_min)
                print("New Model Temperature: {}".format(self.temperature))


            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}".format(epoch, epoch_loss / data.x.shape[0]), end='')
                if y_true is not None:
                    auc = roc_auc_score(y_true, decision_scores)
                    top_k = eval_precision_at_k(y_true, decision_scores, k=y_true.sum())
                    print(f" | AUC {auc:.4f} | top_k {top_k:.4f}", end='')
                print()


        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self


    
    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, edge_index = self.process_graph(sampled_data)

            x_ = self.model(x, edge_index)
            score = self.loss_func(x[:batch_size], x_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = score.detach().cpu().numpy()
        return outlier_scores

    def _data_augmentation(self, x):
        r""" Data augmentation on the input graph. Four types of pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate

        Args:
            x (torch.Tensor): Attribute matrix with dim (n, d).

        Returns:
            tuple: (feat_aug, label_aug)
                    feat_aug is the augmented attribute matrix with dim (n, d),
                    label_aug is the pseudo anomaly label with dim (n,).
        """
        rate = self.r
        surround = self.k
        scale_factor = self.f

        feat_aug = deepcopy(x)
        num_nodes = x.shape[0]
        label_aug = torch.zeros(num_nodes, dtype=torch.int8)

        prob = torch.rand(num_nodes)
        label_aug[prob < rate] = 1

        # deviated
        # a mask of nodes to be deviated
        dv_mask = torch.logical_and(rate / 2 <= prob, prob < rate * 3 / 4)
        # randomly select surrounding nodes
        feat_c = feat_aug[torch.randperm(num_nodes)[:surround]]
        # calculate distance between deviated nodes and surrounding nodes
        ds = torch.cdist(feat_aug[dv_mask], feat_c)
        # assign the least surrounding node to deviated nodes
        feat_aug[dv_mask] = feat_c[torch.argmax(ds, 1)]

        # disproportionate
        # a mask of nodes to be disproportionate with multiple scale factors
        mul_mask = torch.logical_and(rate * 3 / 4 <= prob, prob < rate * 7 / 8)
        # a mask of nodes to be disproportionate with division scale factors
        div_mask = rate * 7 / 8 <= prob
        # scale up or down the attribute of nodes
        feat_aug[mul_mask] *= scale_factor
        feat_aug[div_mask] /= scale_factor

        feat_aug = feat_aug.to(self.device)
        label_aug = label_aug.to(self.device)
        return feat_aug, label_aug

    def _data_augmentation_v2(self, x):
        r""" Data augmentation on the input graph. Four types of pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate

        Args:
            x (torch.Tensor): Attribute matrix with dim (n, d).

        Returns:
            tuple: (feat_aug, label_aug)
                    feat_aug is the augmented attribute matrix with dim (n, d),
                    label_aug is the pseudo anomaly label with dim (n,).
        """
        rate = self.r
        surround = self.k
        scale_factor = self.f

        feat_aug = deepcopy(x)
        num_nodes = x.shape[0]
        label_aug = torch.zeros(num_nodes, dtype=torch.int8)

        prob = torch.rand(num_nodes)
        # label_aug[prob < rate] = 1

        # deviated
        # a mask of nodes to be deviated
        dv_mask = torch.logical_and(rate / 2 <= prob, prob < rate * 3 / 4)
        # randomly select surrounding nodes
        feat_c = feat_aug[torch.randperm(num_nodes)[:surround]]
        # calculate distance between deviated nodes and surrounding nodes
        ds = torch.cdist(feat_aug[dv_mask], feat_c)
        # assign the least surrounding node to deviated nodes
        feat_aug[dv_mask] = feat_c[torch.argmax(ds, 1)]
        label_aug[dv_mask] = 1

        # disproportionate
        # a mask of nodes to be disproportionate with multiple scale factors
        mul_mask = torch.logical_and(rate * 3 / 4 <= prob, prob < rate * 7 / 8)
        # a mask of nodes to be disproportionate with division scale factors
        div_mask = rate * 7 / 8 <= prob
        # scale up or down the attribute of nodes
        feat_aug[mul_mask] *= scale_factor
        feat_aug[div_mask] /= scale_factor
        label_aug[mul_mask] = 1
        label_aug[div_mask] = 1

        feat_aug = feat_aug.to(self.device)
        label_aug = label_aug.to(self.device)
        return feat_aug, label_aug

    def process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        # s = to_dense_adj(G.edge_index)[0].to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, edge_index

    def loss_func(self, x, x_):
        """ Loss function

        :: math::
            L = \\sqrt{\\sum_{i=1}^{n} (x_i - x_i')^2}

        Args:
            x (torch.Tensor): Original attribute matrix with dim (n, d).
            x_ (torch.Tensor): Reconstructed attribute matrix with dim (n, d).

        Returns:
            torch.Tensor: Loss value.
        """
        diff_attribute = torch.pow(x - x_, 2)
        score = torch.sqrt(torch.sum(diff_attribute, 1))
        return score

    def predict(self, G, return_confidence=False):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.

        confidence : numpy array of shape (n_samples,).
            Only if return_confidence is set to True.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        pred_score = self.decision_function(G)
        prediction = (pred_score > self.threshold_).astype(int).ravel()

        if return_confidence:
            confidence = self.predict_confidence(G)
            return prediction, confidence

        return prediction

    def predict_proba(self, G, method='linear', return_confidence=False):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.

        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.

        Returns
        -------
        outlier_probability : numpy array of shape (n_samples, n_classes)
            For each observation, tells whether
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1]. Note it depends on the number of classes, which is by
            default 2 classes ([proba of normal, proba of outliers]).
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(G)

        probs = np.zeros([len(test_scores), 2])

        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]

            if return_confidence:
                confidence = self.predict_confidence(G)
                return probs, confidence

            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]

            if return_confidence:
                confidence = self.predict_confidence(G)
                return probs, confidence

            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

    def predict_confidence(self, G):
        """Predict the model's confidence in making the same prediction
        under slightly different training sets.
        See :cite:`perini2020quantifying`.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        Returns
        -------
        confidence : numpy array of shape (n_samples,)
            For each observation, tells how consistently the model would
            make the same prediction if the training set was perturbed.
            Return a probability, ranging in [0,1].

        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        n = len(self.decision_scores_)

        # todo: this has an optimization opportunity since the scores may
        # already be available
        test_scores = self.decision_function(G)

        count_instances = np.vectorize(
            lambda x: np.count_nonzero(self.decision_scores_ <= x))
        n_instances = count_instances(test_scores)

        # Derive the outlier probability using Bayesian approach
        posterior_prob = np.vectorize(lambda x: (1 + x) / (2 + n))(n_instances)

        # Transform the outlier probability into a confidence value
        confidence = np.vectorize(
            lambda p: 1 - binom.cdf(n - int(n * self.contamination), n, p))(
            posterior_prob)
        prediction = (test_scores > self.threshold_).astype('int').ravel()
        np.place(confidence, prediction == 0, 1 - confidence[prediction == 0])

        return confidence

    def _set_n_classes(self, y):
        """Set the number of classes if `y` is presented, which is not
        expected. It could be useful for multi-class outlier detection.

        Parameters
        ----------
        y : numpy array of shape (n_samples,)
            Ground truth.
        Returns
        -------
        self
        """

        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))
            warnings.warn(
                "y should not be presented in unsupervised learning.")
        return self

    def _process_decision_scores(self):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - labels_: binary labels of training data
        Returns
        -------
        self
        """

        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        # calculate for predict_proba()
        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return


class SSL_Base(nn.Module):
    r""" SSL base model.

    Args:
        in_dim (int): Dimension of input features.
        hid_dim (int): Dimension of hidden layer.
        num_layers (int): Total number of layers, including the decoder layers and encoder layers.
        dropout (float): The dropout rate.
        act (str): The activation function.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout=0.5,
                 model="graphsage",
                 act="relu", 
                 K = 10, 
                 N = 2):
        super(SSL_Base, self).__init__()
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers
        self.K=K
        self.N=N
        self.temperature = 1
        self.shared_encoder = GraphSAGE(in_channels=in_dim,
                                        hidden_channels=hid_dim,
                                        num_layers=encoder_layers,
                                        out_channels=K*N,
                                        dropout=dropout,
                                        act=act)

        self.attr_decoder = GraphSAGE(in_channels=K*N,
                                      hidden_channels=hid_dim,
                                      num_layers=decoder_layers,
                                      out_channels=in_dim,
                                      dropout=dropout,
                                      act=act)

    def embed(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        return h

    def reconstruct(self, h, edge_index):
        # decode attribute matrix
        x_ = self.attr_decoder(h, edge_index)
        return x_

    def forward(self, x, edge_index):
        # encode
        h = self.embed(x, edge_index)
        h = F.log_softmax(h.view(-1, self.N, self.K), dim=-1)            
        # Sampling
        h = sample_gumbel_softmax(h, self.temperature).view(-1, self.N*self.K)
        # reconstruct
        x_ = self.reconstruct(h, edge_index)
        return x_
    


class Merge_PSD_Dataset_v1(InMemoryDataset):
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

        workflows = ["1000genome_new_2022",  "montage"]
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
            print(wf, len(dataset[0]))

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
        for wn in ["1000genome_new_2022",
                   "montage"]:
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
    