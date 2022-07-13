""" Demo of loading dataset in graph-level """

import numpy as np
import torch
from psd_gnn.dataset import PSD_Dataset
from psd_gnn.utils import print_dataset_info

torch.manual_seed(0)
np.random.seed(0)


if __name__ == "__main__":

    workflow = "1000genome"
    ''' load data for binary classification problem in graph level'''
    dataset = PSD_Dataset(root='./',
                          name="1000genome",
                          force_reprocess=False,
                          node_level=False,
                          binary_labels=True)
    print_dataset_info(dataset)

    ''' load data for multi-label classification problem in graph level '''
    dataset = PSD_Dataset(root='./',
                          name="1000genome",
                          force_reprocess=False,
                          node_level=False,
                          binary_labels=False)
    print_dataset_info(dataset)

    ''' load data for binary classification problem in node level '''
    dataset = PSD_Dataset(root='./',
                          name="1000genome",
                          force_reprocess=False,
                          node_level=True,
                          binary_labels=True)
    print_dataset_info(dataset)

    ''' load data for multi-label classification problem in node level '''
    dataset = PSD_Dataset(root='./',
                          name="1000genome",
                          force_reprocess=False,
                          node_level=True,
                          binary_labels=False)
    print_dataset_info(dataset)
