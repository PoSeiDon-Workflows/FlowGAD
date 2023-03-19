# -*- coding: utf-8 -*-
""" A base model for the graph classification.

License: TBD
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap


class GNN(torch.nn.Module):
    def __init__(self,
                 n_node_features: int,
                 n_hidden: int,
                 n_output: int,
                 n_conv_blocks: int = 1) -> None:
        """ Init the GNN model (new version).

        Args:
            n_node_features (int): Number of features at node level.
            n_edge_features (int): Number of features at edge level.
            n_hidden (int): Number of hidden dimension.
            n_output (int): number of output dimension
            n_conv_blocks (int): Number of
        """
        # super class the class structure
        super().__init__()

        # add the ability to add one or more conv layers
        conv_blocks = []

        # ability to add one or more conv blocks
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
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
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

        # NOTE: add the readout layer
        x = gap(x, batch)

        # pass the output to the linear output layer
        out = self.flatten(x)

        # return the output
        return F.log_softmax(out, dim=1)

    def reset_parameters(self):
        """ Reset the parameters of the model."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
