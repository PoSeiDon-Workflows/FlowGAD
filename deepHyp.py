import ray
import json
import pandas as pd
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn


is_gpu_available = torch.cuda.is_available()
n_gpus = torch.cuda.device_count()

print(is_gpu_available)
print(n_gpus)

import torch
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
from torch_geometric.nn import global_mean_pool
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(14, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 4)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    
# flag = 'genome'
# data_read=1
# print(flag)
# if data_read ==0:
#     graphs = load_data(flag)
#     ## Dumped into the pickle file.
#     import pickle
#     with open('graph_all_'+str(flag)+'.pkl','wb') as f:
#         pickle.dump(graphs, f)
# else:
#     import pickle
#     with open('graph_all_'+str(flag)+'.pkl','rb') as f:
#         graphs = pickle.load(f)

import pickle
with open('graph_all_.pkl','rb') as f:
    graphs = pickle.load(f)



import numpy as np
y_list = []
for gr in graphs:
    y_list.append(gr['y'])
print(min(y_list))
print(max(y_list))
print(np.unique(np.array(y_list), return_counts=True))  
datasets=[]
import numpy
for element in graphs:
    gx = torch.tensor(numpy.array(element['x']) ) 
    ge =torch.tensor(numpy.array(element['edge_index']) ).T
    gy =torch.tensor(numpy.array(element['y']).reshape([-1]))
    #print(gx.shape, ge.shape, gy.shape)
    # print(gy)
    v_min, v_max = gx.min(), gx.max()
    new_min, new_max = -1, 1
    gx = (gx - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    # print(gx.min(), gx.max())
    datasets.append( Data(x=gx, edge_index=ge, y=gy) )


import torch
from torch_geometric.datasets import TUDataset
dataset = datasets
torch.manual_seed(12345)
import random
random.seed(12345)
random.shuffle(datasets)
train_dataset = datasets[: int(len(datasets)*0.80) ]
test_dataset = datasets[int(len(datasets)*0.80):]
random.shuffle(train_dataset)
train_dataset= train_dataset[0:2000]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
from torch_geometric.loader import DataLoader


# from IPython.display import Javascript
# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))     
def train(model, criterion, optimizer, loader):
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float(), data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.




def get_run(train_ratio=0.95):
    def run(config: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)

        model = GCN(hidden_channels=int(config["hidden"]) ).float()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        import torch.optim.lr_scheduler as lrs
        # scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
        for _ in range(1, int(config["num_epochs"]) + 1):
            train(model, criterion, optimizer, train_loader)
        accu_test = test(model, test_loader)
        return accu_test
    return run

quick_run = get_run(train_ratio=0.3)
perf_run = get_run(train_ratio=0.95)

# We define a dictionnary for the default values
default_config = {
    "num_epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden":64    
}

# We launch the Ray run-time and execute the `run` function
# with the default configuration

if is_gpu_available:
    if not(ray.is_initialized()):
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)

    run_default = ray.remote(num_cpus=1, num_gpus=1)(perf_run)
    objective_default = ray.get(run_default.remote(default_config))
else:
    if not(ray.is_initialized()):
        ray.init(num_cpus=1, log_to_driver=False)
    run_default = perf_run
    objective_default = run_default(default_config)

print(f"Accuracy Default Configuration:  {objective_default:.3f}")

from deephyper.problem import HpProblem

problem = HpProblem()
# Discrete hyperparameter (sampled with uniform prior)
problem.add_hyperparameter((10, 1000), "num_epochs")
# Discrete and Real hyperparameters (sampled with log-uniform)
problem.add_hyperparameter((8, 128, "log-uniform"), "batch_size")
problem.add_hyperparameter((8, 128, "log-uniform"), "hidden")
problem.add_hyperparameter((0.001, 0.1, "log-uniform"), "learning_rate")
# problem.add_hyperparameter((0.00001, 0.9999, "log-uniform"), "gamma")
# Add a starting point to try first
problem.add_starting_point(**default_config)
problem

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback

def get_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [LoggerCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
    return evaluator

evaluator_1 = get_evaluator(quick_run)
from deephyper.search.hps import AMBS
# Instanciate the search with the problem and a specific evaluator
search = AMBS(problem, evaluator_1)
results = search.search(max_evals=100)