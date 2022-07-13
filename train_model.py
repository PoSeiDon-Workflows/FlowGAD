import argparse
import os.path as osp
import pickle
import random

import numpy as np
import torch
import torch.optim.lr_scheduler as lrs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from psd_gnn.models import GCN
from psd_gnn.utils import parse_data


def load_data(flag, label="all"):
    """ Load data from json file

    Args:
        flag (str): Name of graphs.
        label (str, optional): Label of abnormal scenarios. Defaults to "all".

    Returns:
        dict: Graph in dictionary structure.
    """
    if label == "all":
        classes = {"normal": 0, "cpu": 1, "hdd": 2, "loss": 3}
    elif label == "cpu":
        classes = {"normal": 0, "cpu": 1}
    elif label == "hdd":
        classes = {"normal": 0, "hdd": 1}
    elif label == "loss":
        classes = {"normal": 0, "loss": 1}

    json_path = f"adjacency_list_dags/{flag}.json"
    json_path = json_path.replace("-", "_")
    graphs = parse_data(flag, json_path, classes)
    return graphs


def train(train_loader, model, criterion, optimizer):
    """ Train a Graph Neural Network.

    Args:
        train_loader (pyg.DataLoader): Train dataloader.
        model (object): Model object.
        criterion (function handler): Loss function.
        optimizer (function handler): Optimizer function.
    """
    model.train()

    total_loss = 0
    for data in train_loader:
        # Perform a single forward pass.
        data = data.to(DEVICE)
        optimizer.zero_grad()           # Clear gradients.
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = criterion(out, data.y)   # Compute the loss.
        loss.backward()                 # Derive gradients.
        optimizer.step()                # Update parameters based on gradients.
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


def evaluate(loader, model, testing=False):
    """ Evaluate the GNN model.

    Args:
        loader (pyg.DataLoader): Dataloader to evaluate.
        model (object): Model object.
        testing (bool, optional): Testing case. Defaults to "False".

    Returns:
        float: Accuracy rate.
    """
    model.eval()
    correct = 0
    # Iterate in batches over the training/test dataset.
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x.float(), data.edge_index, data.batch)
        pred = out.argmax(dim=1)                # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)




def train_model(flag, data_read=1, epochs=500):    
    print("Workflow type: {}".format(flag))   
    # parse data from raw files
    if data_read == 0:
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

    print("Number of unique classes: {}".format(max(y_list) - min(y_list) + 1))
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
    
    model     = GCN(hidden_channels=64).float().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(1, epochs+1):
        print("Current epoch: {}".format(epoch))
        train(train_loader, model, criterion, optimizer)
        train_acc = evaluate(train_loader, model)
        test_acc  = evaluate(test_loader, model)
        
        if epoch%10==0:
            scheduler.step()
            print(f'Epoch: {epoch:d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        if epoch%10==0:
            torch.save(model.state_dict(), 'model_'+flag+'.pkl')


type_names = [ "nowcast-clustering-16","1000genome", "nowcast-clustering-8",
              "wind-clustering-casa","wind-noclustering-casa" ]

def main():
	train_model("1000genome", data_read=0, epochs=21)

	return

if __name__ == '__main__':
	main()


