import ray
import json
import pandas as pd
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn
from torch_geometric.loader import DataLoader


def load_data():   
    from torch_geometric.data import Data
    import pickle
    with open('graph_all_.pkl','rb') as f:
        graphs = pickle.load(f)

    import numpy as np
    y_list = []
    for gr in graphs:
        y_list.append(gr['y'])
    # print(min(y_list))
    # print(max(y_list))
    # print(np.unique(np.array(y_list), return_counts=True))  
    datasets=[]
    import numpy
    for element in graphs:
        gx = torch.tensor(numpy.array(element['x']) ) 
        ge =torch.tensor(numpy.array(element['edge_index']) ).T
        gy =torch.tensor(numpy.array(element['y']).reshape([-1]))
        #print(gx.shape, ge.shape, gy.shape)
        # print(gy)
        v_min, v_max = gx.min(), gx.max()
        new_min, new_max = 0, 1
        gx = (gx - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        # print(gx.min(), gx.max())
        datasets.append( Data(x=gx, edge_index=ge, y=gy) )
    return datasets

def run(config: dict):
    import torch
    from torch.nn import Linear
    import torch.nn.functional as F
    from torch_geometric.nn import  SAGEConv, GCNConv
    from torch_geometric.nn import global_mean_pool

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels, drop_rate, hidden, layer_index):
            super(GCN, self).__init__()
            layers = [SAGEConv, GCNConv]
            torch.manual_seed(12345)
            self.conv1 = layers[layer_index](14, hidden_channels).to(device)
            self.conv2=[]
            for i in range(hidden):
                self.conv2.append(layers[layer_index](hidden_channels, hidden_channels).to(device) )
            self.conv3 = layers[layer_index](hidden_channels, hidden_channels).to(device)
            self.drop=drop_rate
            self.lins=[]
            for i in range(hidden):
                self.conv2.append( Linear(hidden_channels, hidden_channels).to(device) )
            self.lin = Linear(hidden_channels, 4).to(device)

        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings 
            x = self.conv1(x, edge_index)
            x = x.relu()

            for i in range(len(self.conv2)):
                x = self.conv2[i](x, edge_index)
                x = x.relu()

            x = self.conv3(x, edge_index)
            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            # 3. Apply a final classifier
            for i in range(len(self.lins)):
                x = self.lins[i](x)
                x = x.relu()
            x = F.dropout(x, p=self.drop, training=self.training)
            x = self.lin(x)
            return x

    # from IPython.display import Javascript
    # display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))     
    def train(model, criterion, optimizer, loader):
        model.train()
        for data in loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
            loss = criterion(out, data.y.to(device))  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

    def test(model, loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x.float().to(device),\
                  data.edge_index.to(device),\
                  data.batch.to(device))  
            pred = out.argmax(dim=1).detach().cpu()  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
        
    datasets = load_data()
    torch.manual_seed(12345)
    import random
    random.seed(12345)
    random.shuffle(datasets)
    train_dataset = datasets[: int(len(datasets)*0.60) ]
    test_dataset = datasets[int(len(datasets)*0.60):int(len(datasets)*0.80)]
    random.shuffle(train_dataset)
    train_dataset= train_dataset[0:200]

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)
    model = GCN(hidden_channels=int(config["hidden"]), drop_rate = config["dropout"],\
                hidden = int(config["hiddenlayer"]),\
                layer_type=int(config["layer_type"]) ).float().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])


    # import torch.optim.lr_scheduler as lrs
    # scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
    for _ in range(1, 2000):     
        train(model, criterion, optimizer, train_loader)
    
    accu_test = test(model, test_loader)
    return accu_test

# # We launch the Ray run-time and execute the `run` function
# # with the default configuration
# if is_gpu_available:
#     if not(ray.is_initialized()):
#         ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)

#     run_default = ray.remote(num_cpus=1, num_gpus=1)(perf_run)
#     objective_default = ray.get(run_default.remote(default_config))
# else:
#     if not(ray.is_initialized()):
#         ray.init(num_cpus=1, log_to_driver=False)
#     run_default = perf_run
#     objective_default = run_default(default_config)



def get_evaluator(run_function):
    from deephyper.evaluator.callback import LoggerCallback
    is_gpu_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "address":'auto',
        "num_cpus_per_task": 1,
        "callbacks": [LoggerCallback()]
    }
    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    # print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
    return evaluator


def problem():
    # We define a dictionnary for the default values
    default_config = {
        "hiddenlayer": 2,
        "batch_size": 62,
        "dropout":  0.4,
        "learning_rate": 0.001,
        "hidden":125,
        "layer_type":0,
    }
    from deephyper.problem import HpProblem
    problem = HpProblem()
    # Discrete hyperparameter (sampled with uniform prior)
    # problem.add_hyperparameter((10, 1000), "num_epochs")
    # Discrete and Real hyperparameters (sampled with log-uniform)
    problem.add_hyperparameter((0, 1, "uniform"), "layer_type")
    problem.add_hyperparameter((1, 4, "uniform"), "hiddenlayer")
    problem.add_hyperparameter((8, 128, "uniform"), "batch_size")
    problem.add_hyperparameter((8, 128, "uniform"), "hidden")
    problem.add_hyperparameter((0.001, 0.1, "log-uniform"), "learning_rate")
    problem.add_hyperparameter((0.01, 1, "uniform"), "dropout")
    # problem.add_hyperparameter((0.00001, 0.9999, "log-uniform"), "gamma")
    # Add a starting point to try first
    problem.add_starting_point(**default_config)
    return problem


if __name__ == "__main__":
    from deephyper.search.hps import AMBS
    from deephyper.evaluator import Evaluator
    prob1 = problem()
    evaluator_1= get_evaluator(run)
    print("the total number of deep hyper workers are", evaluator_1.num_workers) 
    # Instanciate the search with the problem and a specific evaluator
    search = AMBS(prob1, evaluator_1)
    search.search(max_evals=200)