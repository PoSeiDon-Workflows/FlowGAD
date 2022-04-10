import ray
import json
import pandas as pd
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn
from torch_geometric.loader import DataLoader


def update_CL_(model, optimizer, criterion, mem_loader, train_loader, task, Graph = 0,\
     params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
               'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
              'batchsize': 64, 'total_updates': 1000 } ):
    device = params['device']
    def normalize_grad(input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
    import copy
    # We set up the iterators for the memory loader and the train loader
    mem_iter = iter(mem_loader)
    task_iter = iter(train_loader)
    
    # The main loop over all the batch
    for i in range(params['total_updates']): 
        if task>0:
            ###########################################
            ## this is for when graph or not graph
            if Graph ==0:
                # Extract a batch from the task
                try:
                    data_t= next(task_iter)
                    if data_t.y.shape[0]<params['batchsize']:
                        task_iter = iter(train_loader)
                        data_t = next(task_iter)
                except StopIteration:
                    task_iter = iter(train_loader)
                    data_t= next(task_iter)

                # Extract a batch from the memory
                try:
                    data_m= next(mem_iter)
                    if data_m.y.shape[0]<params['batchsize']:
                        mem_iter = iter(mem_loader)
                        data_m= next(mem_iter)    
                except StopIteration:
                    mem_iter = iter(mem_loader)
                    data_m= next(mem_iter)
            else:
                 # Extract a batch from the task
                try:
                    data_t= next(task_iter)
                    (_, y) = data_t
                    if y.shape[0]<params['batchsize']:
                        task_iter = iter(train_loader)
                        data_t = next(task_iter)
                except StopIteration:
                    task_iter = iter(train_loader)
                    data_t= next(task_iter)
                
                
                # Extract a batch from the memory
                try:
                    data_m= next(mem_iter)
                    (_, y) = data_m
                    if y.shape[0]<params['batchsize']:
                        mem_iter = iter(mem_loader)
                        data_m= next(mem_iter)    
                except StopIteration:
                    mem_iter = iter(mem_loader)
                    data_m= next(mem_iter)
            ###########################################

            
            ###########################################
            ## For GRAPHS
            if Graph ==0:
                # print("Graphs")
                # Send the data to the device
                data_m = data_m.to(device)
                data_t = data_t.to(device)
                # Apply the model on the task batch and the memory batch
                out = model(data_t.x.float().to(device), data_t.edge_index.to(device), data_t.batch.to(device))  # Perform a single fo
                out_m = model(data_m.x.float().to(device), data_m.edge_index.to(device), data_m.batch.to(device))
                
                ## Get loss on the memory and task and put it together
                J_P = criterion(out, data_t.y.to(device))
                J_M = criterion(out_m, data_m.y.to(device))

                
                ############## This is the J_x loss
                #########################################################################################
                # Add J_x  now
                x_PN = copy.copy(data_m.x).to(device)
                x_PN.requires_grad = True
                adv_grad = 0
                epsilon = params['x_lr']
                # The x loop
                for epoch in range(params["x_updates"]):
                    # The datapoint
                    x_PN = x_PN+ epsilon*adv_grad
                    crit = criterion(model(x_PN.float(), data_m.edge_index, data_m.batch), data_m.y)
                    loss = torch.mean(crit) + torch.var(crit)
                    # Calculate the gradient
                    adv_grad = torch.autograd.grad( loss,x_PN)[0]
                    # Normalize the gradient values.
                    adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                
                # The critical cost function
                J_x_crit = (criterion(model(x_PN.float(), data_m.edge_index, data_m.batch), data_m.y) \
                         -  criterion(model(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y) )

            
                ############### This is the loss J_th
                #########################################################################################
                cop = copy.deepcopy(model).to(device)
                opt_buffer = torch.optim.Adam(cop.parameters(),lr = params['th_lr'])
                J_PN_theta = criterion(model(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y)
                for i in range(params["theta_updates"]):
                    opt_buffer.zero_grad()
                    loss_crit = criterion(cop(data_t.x.float(), data_t.edge_index, data_t.batch), data_t.y)
                    loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
                    #+ torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
                    loss_m.backward(retain_graph=True)
                    opt_buffer.step()
                J_th_crit = criterion(cop(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y) - J_PN_theta


                # Now, put together  the loss fully 
                Total_loss= torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit+J_th_crit) \
                          + torch.var(params['factor']*J_P+J_M+J_x_crit+params['factor']*J_th_crit)
                
            
            ## For REGULAR NET
            else:
                in_t, targets_t= data_t
                in_m, targets_m = data_m
                
                
                in_t = in_t.unsqueeze(dim=1).float().to(device)
                in_m = in_m.unsqueeze(dim=1).float().to(device)
                targets_t=targets_t.to(device)
                targets_m=targets_m.to(device)
                
                
                out = model(in_t)
                out_m = model(in_m)
                
                ############## The task cost and the memory cost
                #########################################################################################
                J_P = criterion(out, targets_t.to(device))
                J_M = criterion(out_m, targets_m.to(device))

                ############## This is the J_x loss
                #########################################################################################
                J_PN_x=criterion(model(in_m), targets_m)
                x_PN = copy.copy(in_m).to(device)
                x_PN.requires_grad = True
                adv_grad = 0
                epsilon =params['x_lr']
                
                
                for epoch in range(params["x_updates"]):
                    x_PN = x_PN+ epsilon*adv_grad
                    crit = criterion(model(x_PN.float() ), targets_m)
                    loss = torch.mean(crit) + torch.var(crit) # + skew(crit) + kurtosis(crit)
                    adv_grad = torch.autograd.grad(loss,x_PN)[0]
                    # Normalize the gradient values.
                    adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                J_x_crit = (criterion(model(x_PN.float()), targets_m) -J_PN_x)

                ############### This is the loss J_th
                #########################################################################################
                cop = copy.deepcopy(model).to(device)
                opt_buffer = torch.optim.Adam(cop.parameters(),lr = params['th_lr'])
                J_PN_theta = criterion(model(in_m.float()), targets_m)
                for i in range(params["theta_updates"]):
                    opt_buffer.zero_grad()
                    loss_crit = criterion(cop(in_t.float()), targets_t)
                    loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
                    #+ torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
                    loss_m.backward(retain_graph=True)
                    opt_buffer.step()
                J_th_crit = (criterion(cop(in_m.float()), targets_m) - J_PN_theta)
                
                # Now, put together  the loss fully 
                Total_loss= torch.mean(J_M+J_P+params['factor']*J_x_crit+params['factor']*J_th_crit)
                # +torch.var(J_P+J_M+J_x_crit+params['factor']*J_th_crit)
                
                
            optimizer.zero_grad()
            Total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            
        ## FOR when there is only one task
        else:
 
            if Graph==0:
                 # Extract a batch from the task
                try:
                    data_t= next(task_iter)
                    if data_t.y.shape[0]<params['batchsize']:
                        task_iter = iter(train_loader)
                        data_t = next(task_iter)
                except StopIteration:
                    task_iter = iter(train_loader)
                    data_t= next(task_iter)
                    
                out = model(data_t.x.float().to(device), data_t.edge_index.to(device), data_t.batch.to(device))  # Perform a single forward pass.
                critti= criterion(out, data_t.y.to(device))
                Total_loss = torch.mean(critti)+ torch.var(critti)
                optimizer.zero_grad() 
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
            else:
                 # Extract a batch from the task
                try:
                    data_t= next(task_iter)
                    (_, y) = data_t
                    if y.shape[0]<params['batchsize']:
                        task_iter = iter(train_loader)
                        data_t = next(task_iter)
                except StopIteration:
                    task_iter = iter(train_loader)
                    data_t= next(task_iter)
                    
                in_t, targets_t = data_t 
                in_t = in_t.unsqueeze(dim=1).float()
                
                critti= criterion(model(in_t.to(device)), targets_t.to(device))
                Total_loss = torch.mean(critti) + torch.var(critti)
                optimizer.zero_grad()
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
       
    ## Return all the data back.
    if task>0:
        return Total_loss.detach().cpu(),\
              (torch.mean(J_M+ params['factor']*J_x_crit)+torch.var(J_P+J_M+params['factor']*J_x_crit+params['factor']*J_th_crit)).detach().cpu(),\
              (torch.mean(J_P+params['factor']*J_th_crit)+torch.var(J_P+J_M+params['factor']*J_x_crit+params['factor']*J_th_crit)).detach().cpu()
    else:
        return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()


def load_data(file):   
    from torch_geometric.data import Data
    # nowcast clustering
    # Import the file.
    import pickle
    with open('pickles/'+file+'.pkl','rb') as f:
        graphs= pickle.load(f) 
    # Separate the data according to labels
    # One for every type of workflow
    import numpy as np
    y_list = []
    for gr in graphs:
        y_list.append(gr['y'])
    tot = np.unique(np.array(y_list), return_counts=True)[1][1]
    import torch
    from torch_geometric.data import Data
    datasets=[]

    for element in graphs:
        gx = torch.tensor(np.array(element['x']) ) 
        ge =torch.tensor(np.array(element['edge_index']) ).T
        gy =torch.tensor(np.array(element['y']).reshape([-1]))
        v_min, v_max = gx.min(), gx.max()
        new_min, new_max = 0, 1
        gx = (gx - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        # print(gx.min(), gx.max())
        datasets.append( Data(x=gx, edge_index=ge, y=gy) )
    
    return datasets


def run_test(config: dict):
    return config['hidden']        


def run(config: dict):
    import torch
    from torch.nn import Linear
    import torch.nn.functional as F
    from torch_geometric.nn import  SAGEConv, GCNConv
    from torch_geometric.nn import global_mean_pool
    import random
    from torch_geometric.loader import DataLoader
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


    ########################## Define the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class GCN(torch.nn.Module):

        def __init__(self, hidden_channels, drop_rate, hidden, layer_index):
            super(GCN, self).__init__()
            layers = SAGEConv
            torch.manual_seed(12345)
            self.conv1 = SAGEConv(14, hidden_channels).to(device)
            self.conv2=[]
            for i in range(hidden):
                self.conv2.append(layers(hidden_channels, hidden_channels).to(device) )
            self.conv3 = layers(hidden_channels, hidden_channels).to(device)

            self.drop=drop_rate
            self.lins= Linear(hidden_channels, hidden_channels).to(device)
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
            x = self.lins(x)
            x = x.relu()
            x = F.dropout(x, p=self.drop, training=self.training)
            
            x = self.lin(x)
            
            return x


    memory_train=[]
    memory_valid=[]
    filenames =['graph_all_casa_nowcast_8', 'graph_all_genome',\
                'graph_all_nowcluster_16', 'graph_all_wind_clustering', \
                'graph_all_wind_noclustering']

    is_gpu_available = torch.cuda.is_available()
    import os
    print("GPU is available? ", is_gpu_available, os.environ.get("CUDA_VISIBLE_DEVICES"))
    model = GCN(hidden_channels=int(config["hidden"]),\
                drop_rate = config["dropout"],\
                hidden = int(config["hiddenlayer"]),\
                layer_index=0 ).float().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    for (i,file) in enumerate(filenames):
        # print("new task", file)
        datasets = load_data(file)
        torch.manual_seed(12345)
        random.seed(12345)
        lengtha=len(datasets)
        random.shuffle(datasets)
        train_dataset = datasets[:int(0.60*lengtha)]
        valid_dataset = datasets[int(0.60*lengtha):int(0.80*lengtha)]
        memory_train+=train_dataset
        memory_valid+=valid_dataset

        train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
        mem_train_loader = DataLoader(memory_train, batch_size=int(config["batch_size"]), shuffle=True)
        import time
        start_= time.time()
        for e in range(1,500):
            # print("The epoch", e, "for filename", file, "with", len(datasets),\
            #      "and took", time.time()-start_ )
            _, _, _ = update_CL_( model, optimizer, criterion,\
            mem_train_loader, train_loader, task=i,\
            params = {'x_updates': int(config['x_updates']),\
            'theta_updates': int(config['theta_updates']),\
            'factor': config['factor'],\
            'x_lr':  config["learning_rate"]*0.001,\
            'th_lr': config["learning_rate"]*0.001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize': int(config['batch_size']),\
            'total_updates': int(config['total_updates']) }  )

    mem_test_loader = DataLoader(memory_valid, batch_size=int(config["batch_size"]), shuffle=True)
    accu_test = test(model, mem_test_loader)
    print(" The evaluation took", time.time()-start_ , "and provided", accu_test)
    return accu_test


def get_evaluator(run_function):
    from deephyper.evaluator.callback import LoggerCallback
    is_gpu_available = torch.cuda.is_available()
    import os
    print("GPU is available? ", is_gpu_available, os.environ.get("CUDA_VISIBLE_DEVICES"))
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
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
    return evaluator



def problem():
    # We define a dictionnary for the default values
    default_config = {
        "hiddenlayer": 2,
        "batch_size": 64,
        "dropout":  0.5,
        "learning_rate": 0.001,
        "hidden":125,
        'x_updates': 3,
        'theta_updates':3, 
        'factor': 0.99,
        'total_updates': 100}

    from deephyper.problem import HpProblem
    problem = HpProblem()
    problem.add_hyperparameter((1, 4, "uniform"), "hiddenlayer")
    problem.add_hyperparameter((8, 128, "uniform"), "batch_size")
    problem.add_hyperparameter((8, 128, "uniform"), "hidden")
    problem.add_hyperparameter((0.001, 0.1, "log-uniform"), "learning_rate")
    problem.add_hyperparameter((0.01, 1, "log-uniform"), "dropout")
    ## The hyper-parameters for the continual learning part.
    problem.add_hyperparameter((1, 5, "uniform"), "theta_updates")
    problem.add_hyperparameter((1, 5, "uniform"), "x_updates")
    problem.add_hyperparameter((0.000001, 1, "uniform"), "factor")
    problem.add_hyperparameter((50, 1200, "uniform"), "total_updates")
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
    search.search(max_evals=20)