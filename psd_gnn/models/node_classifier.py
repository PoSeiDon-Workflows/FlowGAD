# -*- coding: utf-8 -*-
""" A base model for the node classification.

License: TBD
Date: 2022-06-30
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GCNConv, SAGEConv
torch.manual_seed(0)


class GNN(torch.nn.Module):
    def __init__(self,
                 n_node_features: int,
                 n_hidden: int,
                 n_output: int,
                 n_conv_blocks: int = 1) -> None:
        """ Init the GNN model (new version).

        Args:
            n_node_features (int): Number of features at node level.
            n_hidden (int): Number of hidden dimension.
            n_output (int): number of output dimension
            n_conv_blocks (int): Number of
        """
        # super class the class structure
        super().__init__()

        # add the ability to add one or more conv layers
        conv_blocks = []

        # ability to  add one or more conv blocks
        for _ in range(n_conv_blocks):
            conv_blocks += [
                GCNConv(n_node_features, n_hidden),
                ReLU(),
                GCNConv(n_hidden, n_hidden),
                ReLU(),
            ]

        # group all the conv layers
        self.conv_layers = ModuleList(conv_blocks)

        # add the linear layers for flattening the output from MPNN
        self.flatten = Sequential(
            Linear(n_hidden, n_hidden),
            ReLU(),
            Linear(n_hidden, n_output))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """ Processing the GNN model.

        Args:
            x (torch.Tensor): Input features at node level.
            edge_index (torch.Tensor): Index pairs of vertices

        Returns:
            torch.Tensor: output tensor.
        """
        # process the layers
        for layer in self.conv_layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)

        # pass the output to the linear output layer
        out = self.flatten(x)

        # return the output
        return F.log_softmax(out, dim=1)


class GNN_v2(torch.nn.Module):
    """ A GraphSage based model (old version) """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GNN_v2, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # NOTE: without global pooling layer

        # 2. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x
