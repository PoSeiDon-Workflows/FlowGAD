""" Base model for generative model for synthetic anomalies. """
import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch_geometric.nn import MLP, GCNConv
from torch_geometric.nn import global_mean_pool as gap


class Gen(Module):
    """ Generator """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers=1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.mlp = MLP(in_channels=self.in_dim,
                       hidden_channels=self.hid_dim,
                       out_channels=self.out_dim,
                       num_layers=self.num_layers)

    def forward(self, noise):
        """ Forward pass on generator.

        Args:
            noise (torch.Tensor): Noise input with dim (n, m).

        Returns:
            torch.Tensor: Output tensor with same dim as input.
        """
        return self.mlp(noise)


class Dis(Module):
    """ Discriminator """

    def __init__(self, in_dim, hid_dim, out_dim=2, num_layers=2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.encoder = GCNConv(in_channels=self.in_dim,
                               out_channels=self.hid_dim)

        self.dis = MLP(in_channels=1,
                       hidden_channels=self.hid_dim,
                       out_channels=self.out_dim,
                       num_layers=self.num_layers)

    def forward(self, x, edge_index, batch):
        """ Forward pass on discriminator.

        Args:
            x (torch.tensor): Node feature matrix with dim (n, d).
            edge_index (torch.tensor): Pair of edge index with dim (m, 2).

        Returns:
            torch.tensor: Output after discriminator.
        """
        z = self.encoder(x, edge_index).relu()
        z = gap(z, batch=batch)
        return self.dis(z).sigmoid()


def anomaly_score(x_real, x_fake):
    torch.linalg.norm(x_real - x_fake, dim=1)


def weights_init(m):
    # REVIEW: problem in `apply` function
    """ Initialize the weights in G and D.

    Args:
        m (object): Model Object.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
