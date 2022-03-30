import json
import glob
import os
import pandas as pd
import numpy as np
import pickle
import random

import torch
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
import torch.optim.lr_scheduler as lrs
from torch_geometric.loader import DataLoader

from helpers.parsing_functions import parse_data
from helpers.models import GCN

torch.manual_seed(12345)
random.seed(12345)


def load_data(flag):

    classes = {"normal": 0}
    counter = 1
    json_path = ""
    
    for d in os.listdir("data"):
        d = d.split("_")[0]
        
        if d in classes:
            continue
        classes[d] = counter
        
        counter += 1
        
    if flag == "nowcast-clustering-16":
        json_path = "adjacency_list_dags/casa_nowcast_clustering_16.json"
    elif flag == "1000genome":
        json_path = "adjacency_list_dags/1000genome.json"
    elif flag =="nowcast-clustering-8":
        json_path = "adjacency_list_dags/casa_nowcast_clustering_8.json"
    elif flag == "wind-clustering-casa":
        json_path = "adjacency_list_dags/casa_wind_clustering.json"
    elif flag == "wind-noclustering-casa":
        json_path = "adjacency_list_dags/casa_wind_no_clustering.json"
        
    graphs = parse_data(flag, json_path, classes)
    return graphs




def train(train_loader, model, criterion, optimizer):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.



def test(loader, model):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float(), data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.




def train_mod(flag, data_read=1, epochs=500):    
    print(flag)
    
    # parse data from raw files
    if data_read ==0:
        graphs = load_data(flag)
        print(len(graphs))
        with open('graph_all_'+ str(flag) + '.pkl','wb') as f:
            pickle.dump(graphs, f)
    else:
        with open('graph_all_'+ str(flag) + '.pkl','rb') as f:
            graphs = pickle.load(f)
    
    y_list = []
    for gr in graphs:
        y_list.append(gr['y'])
    print(min(y_list))
    print(max(y_list))
    print(np.unique(np.array(y_list), return_counts=True))  
    datasets=[]

    for element in graphs:
        gx = torch.tensor(np.array(element['x']) ) 
        ge = torch.tensor(np.array(element['edge_index']) ).T
        gy = torch.tensor(np.array(element['y']).reshape([-1]))

        v_min, v_max = gx.min(), gx.max()
        new_min, new_max = -1, 1
        gx = (gx - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        datasets.append( Data(x=gx, edge_index=ge, y=gy) )

    dataset = datasets
    random.shuffle(datasets)
    train_dataset = datasets[: int(len(datasets)*0.80) ]
    test_dataset  = datasets[int(len(datasets)*0.80):]
    
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = GCN(hidden_channels=64).float()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(1, epochs):
        print(epoch)
        train(train_loader, model, criterion, optimizer)
        train_acc = test(train_loader, model)
        test_acc  = test(test_loader, model)
        
        if epoch%100==0:
            scheduler.step()
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        if epoch%100==0:
            torch.save(model.state_dict(), 'model_'+flag+'.pkl')


type_names = [ "nowcast-clustering-16","1000genome", "nowcast-clustering-8",
              "wind-clustering-casa","wind-noclustering-casa" ]

def main():
	train_mod('1000genome', data_read=0, epochs=300)

	return

if __name__ == '__main__':
	main()


