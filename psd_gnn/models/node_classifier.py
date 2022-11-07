# -*- coding: utf-8 -*-
""" A base model for the node classification.

License: TBD
Date: 2022-06-30
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential, ELU
from torch_geometric.nn import GCNConv, GATConv

torch.manual_seed(0)


class GNN(torch.nn.Module):
    def __init__(self,
                 n_node_features: int,
                 n_hidden: int,
                 n_output: int,
                 n_conv_blocks: int = 1,
                 dropout:float = 0.5) -> None:
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
            # REVIEW: with attention layers
            # conv_blocks += [
            #     GATConv(n_node_features, 8, heads=8, dropout=0.5),
            #     ELU(),
            #     GATConv(8 * 8, n_hidden, heads=1, concat=False, dropout=0.5),
            # ]

        # group all the conv layers
        self.conv_layers = ModuleList(conv_blocks)

        self.dropout = dropout
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
            if isinstance(layer, GCNConv) or isinstance(layer, GATConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass the output to the linear output layer
        out = self.flatten(x)

        # return the output
        return F.log_softmax(out, dim=1)
