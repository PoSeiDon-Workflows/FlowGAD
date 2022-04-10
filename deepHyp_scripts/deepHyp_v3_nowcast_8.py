import ray
import json
import pandas as pd
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn
from torch_geometric.loader import DataLoader

import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,  SAGEConv
from torch_geometric.nn import global_mean_pool
    
def convert_tensors(graphs):
    import numpy as np
    y_list = []
    for gr in graphs:
        y_list.append(gr['y'])
    import torch
    from torch_geometric.data import Data
    datasets=[]
    import numpy
    from sklearn.preprocessing import StandardScaler
    scaler =  StandardScaler()
    for element in graphs:
        gx = torch.from_numpy(element['x'] )
        ge =torch.tensor(numpy.array(element['edge_index']) ).T
        gy =torch.tensor(numpy.array(element['y']).reshape([-1]))
        if gx.shape[0] >0 :
            datasets.append( Data(x=gx, edge_index=ge, y=gy) )
    # import torch
    # from torch_geometric.datasets import TUDataset
    # print('====================')
    # print(f'Number of graphs: {len(datasets)}')
    # # print(f'Number of features: {dataset.num_features}')
    # # print(f'Number of classes: {dataset.num_classes}')
    # data = datasets[0]  # Get the first graph object.
    # print()
    # print(data)
    # print('=============================================================')
    # # Gather some statistics about the first graph.
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Has self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')
    return datasets


def convert_dataloader(datasets,  batch_size):
    import torch
    import random
    torch.manual_seed(12345)
    random.seed(12345)
    random.shuffle(datasets)
    
    train_dataset = datasets[: int(len(datasets)*0.80) ]
    test_dataset = datasets[int(len(datasets)*0.80):]
    # print(f'Number of training graphs: {len(train_dataset)}')
    # print(f'Number of test graphs: {len(test_dataset)}')
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


import torch
def process_data(graphs, drop_columns):## Now, preprocess that for the columns I specify
    import numpy as np
    import pandas as pd
    # Get all
    import json
    import glob
    import os
    import pandas as pd
    import numpy as np
    classes = {"normal": 0}
    counter = 1
    for d in os.listdir("data"):
        d = d.split("_")[0]
        if d in classes: continue
        classes[d] = counter
        counter += 1
    columns_full= ['type', 'is_clustered', 'ready',
        'submit', 'execute_start', 'execute_end', 'post_script_start',
        'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
        'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay',
        'stage_in_bytes', 'stage_out_bytes', 'kickstart_user', 'kickstart_site', 
        'kickstart_hostname', 'kickstart_transformations', 'kickstart_executables',
        'kickstart_executables_argv', 'kickstart_executables_cpu_time', 'kickstart_status',
        'kickstart_executables_exitcode']

    number_data=[]
    flat_list = [item  for sublist in graphs for item in sublist['x']]
    number_data = [len(sublist['x']) for sublist in graphs]
    df = pd.DataFrame(flat_list, columns=columns_full)
    df = df.drop(drop_columns, axis=1)

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    for string in ['type', 'kickstart_user', 'kickstart_site',\
                   'kickstart_hostname', 'kickstart_transformations']:
            df[string] =labelencoder.fit_transform(df[string].astype(str))
    
    array = df.to_numpy().astype('float64')
    prev = 0
    nexts= 0
    new_min, new_max = 0, 1
    gx = array
    v_min, v_max = gx.min(), gx.max()
    gx = (gx - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    # print(gx.min(), gx.max())
    from sklearn.preprocessing import StandardScaler
    scaler =  StandardScaler()
    scaler.fit_transform(gx)
    for i in range(len(graphs)):
        nexts+=number_data[i]
        graphs[i]['x']= array[prev:nexts,:]
        prev+=number_data[i]
    # print(len(graphs))
    return graphs

## The main code for the deephyper run...
def load_data():   
    import pickle
    with open('graph_nowcluster_8_preprocessed.pkl','rb') as f:
        graphs = pickle.load(f)
    drop_columns= [ 'stage_in_bytes', 'stage_out_bytes', 'kickstart_executables','kickstart_executables_argv']
    graphs = process_data(graphs, drop_columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets= convert_tensors(graphs)
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
            self.conv1 = layers[layer_index](22, hidden_channels).to(device)
            self.conv2=[]
            for i in range(hidden):
                self.conv2.append(layers[layer_index](hidden_channels, hidden_channels).to(device) )
            self.conv3 = layers[layer_index](hidden_channels, hidden_channels).to(device)
            self.drop=drop_rate
            self.lins=[]
            for i in range(hidden):
                self.lins.append( Linear(hidden_channels, hidden_channels).to(device) )
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

    def train(model,  criterion, optimizer, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.train()
        for data in loader:  # Iterate in batches over the training dataset.
            out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
            loss = criterion(out, data.y.to(device))  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(model, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        correct = 0            
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    datasets = load_data()
    torch.manual_seed(12345)
    import random
    random.seed(12345)

    train_loader, test_loader = convert_dataloader(datasets, batch_size=32)
    model = GCN(hidden_channels=int(config["hidden"]), drop_rate = config["dropout"],\
                hidden = int(config["hiddenlayer"]),\
                layer_index=int(config["layer_type"])).float().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    for epoch in range(1, 300):     
        train(model, criterion, optimizer, train_loader)
        # train_acc = test(model, train_loader)
        # test_acc = test(model, test_loader)
        # if epoch%100==0:
        #     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        #     flag='genome'   
        #     # torch.save(model.state_dict(), 'model_'+flag+'_')
    accu_test = test(model, test_loader)
    return accu_test


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
        "batch_size": 32,
        "dropout":  0.1,
        "learning_rate": 0.001,
        "hidden":32,
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
    # Instantiate the search with the problem and a specific evaluator
    search = AMBS(prob1, evaluator_1)
    search.search(max_evals=1000)