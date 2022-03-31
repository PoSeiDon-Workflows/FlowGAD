import torch
def update_CL_(model, criterion, mem_loader, train_loader, task, Graph = 0,\
     params = {'x_updates': 2,  'theta_updates':2, 'factor': 0.6, 'x_lr': 0.0001,'th_lr':0.001,\
               'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
              'batchsize': 64 } ):
    device = params['device']
    def normalize_grad(input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
    import copy
    
    
    # We set up the iterators for the memory loader and the train loader
    mem_iter = iter(mem_loader)
    task_iter = iter(train_loader)
    
    # The main loop over all the batch
    for i in range(100): 
      
        # Extract a batch from the task
        try:
            data_t= next(task_iter)
            if data_t.y.shape[0]<params['batchsize']:
                task_iter = iter(train_loader)
                data_t = next(task_iter)
        except StopIteration:
            task_iter = iter(train_loader)
            data_t= next(mem_iter)
            
        if task>0:
            # Extract a batch from the memory
            try:
                data_m= next(mem_iter)
                if data_m.y.shape[0]<params['batchsize']:
                    mem_iter = iter(mem_loader)
                    data_m= next(mem_iter)    
            except StopIteration:
                mem_iter = iter(mem_loader)
                data_m= next(mem_iter)

                
            # Send the data to the device
            data_m = data_m.to(device)
            data_t = data_t.to(device)
            
            
            ## For GRAPHS
            if Graph ==0:
                
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
                for epoch in range(2):
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
                for i in range(2):
                    opt_buffer.zero_grad()
                    loss_crit = criterion(cop(data_t.x.float(), data_t.edge_index, data_t.batch), data_t.y)
                    loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
                    #+ torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
                    loss_m.backward(retain_graph=True)
                    opt_buffer.step()
                J_th_crit = criterion(cop(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y) - J_PN_theta


                # Now, put together  the loss fully 
                Total_loss= torch.mean(J_M+ J_x_crit)\
                          + params['factor']*torch.mean(J_P+J_th_crit) \
                          + torch.var(params['factor']*J_P+J_M+J_x_crit+params['factor']*J_th_crit)
                
            
            ## For REGULAR NET
            else:
                in_t, targets_t= data_t
                in_m, targets_m = data_m
                
                
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
                
                
                for epoch in range(2):
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
                for i in range(2):
                    opt_buffer.zero_grad()
                    loss_crit = criterion(cop(in_t.float()), targets_t)
                    loss_m = torch.mean(loss_crit) + torch.var(loss_crit)
                    #+ torch.var(loss_crit) + skew(loss_crit) + kurtosis(loss_crit)
                    loss_m.backward(retain_graph=True)
                    opt_buffer.step()
                
                J_th_crit = (criterion(cop(in_m.x.float()), targets_m) - J_PN_theta)
                # Now, put together  the loss fully 
                Total_loss= torch.mean(J_M+ J_x_crit)\
                          + params['factor']*torch.mean(J_P+J_th_crit) \
                          + torch.var(params['factor']*J_P+J_M+J_x_crit+params['factor']*J_th_crit)
                
                
            optimizer.zero_grad()
            Total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            
        ## FOR when there is only one task
        else:
            if Graph==0:
                out = model(data_t.x.float().to(device), data_t.edge_index.to(device), data_t.batch.to(device))  # Perform a single forward pass.
                critti= criterion(out, data_t.y.to(device))
                Total_loss = torch.mean(critti)+ torch.var(critti)
                optimizer.zero_grad() 
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
            else:
                critti= criterion(model(in_t), targets_t.y.to(device))
                Total_loss = torch.mean(critti) + torch.var(critti)
                optimizer.zero_grad()
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.

    ## Return all the data back.
    if task>0:
        return Total_loss.detach().cpu(),\
                (torch.mean(J_M+ J_x_crit)+torch.var(params['factor']*J_P+J_M+J_x_crit+params['factor']*J_th_crit)).detach().cpu(),\
                (params['factor']*torch.mean(J_P+J_th_crit)+torch.var(params['factor']*J_P+J_M+J_x_crit+J_th_crit)).detach().cpu().detach().cpu()
    else:
        return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()




from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
from torch_geometric.nn import global_mean_pool

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.
    return (correct / len(loader.dataset)) # Derive ratio of correct predictions.



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(14, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 4)
        

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
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.0001, training=self.training)
        x = self.lin2(x)
        return x

memory_train=[]
memory_test=[]
memory_valid=[]
filenames =['graph_all_casa_nowcast_8', 'graph_all_genome',\
            'graph_all_nowcluster_16', 'graph_all_wind_noclustering',\
            'graph_all_wind_clustering']
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(hidden_channels=64).float().to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#import torch.optim.lr_scheduler as lrs
#scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
accuracies_mem = []
accuracies_one=[]
Total_loss=[]
Gen_loss=[]
For_loss=[]
for (i,file) in enumerate(filenames):
    print("new task", file)
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
    import numpy
    from sklearn.preprocessing import StandardScaler
    for element in graphs:
        gx = torch.tensor(numpy.array(element['x']) ) 
        ge =torch.tensor(numpy.array(element['edge_index']) ).T
        gy =torch.tensor(numpy.array(element['y']).reshape([-1]))
        v_min, v_max = gx.min(), gx.max()
        new_min, new_max = 0, 1
        gx = (gx - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        # print(gx.min(), gx.max())
        datasets.append( Data(x=gx, edge_index=ge, y=gy) )
        
        
    import random
    torch.manual_seed(12345)
    random.seed(12345)
    lengtha=len(datasets)
    random.shuffle(datasets)
    train_dataset = datasets[:int(0.60*lengtha)]
    valid_dataset = datasets[int(0.60*lengtha):int(0.80*lengtha)]
    test_dataset = datasets[int(0.80*lengtha):]
    
    memory_train+=train_dataset
    memory_valid+=valid_dataset
    memory_test+=test_dataset
    
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of valid graphs: {len(valid_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    
    print(f'Memory:  Number of training graphs: {len(memory_train)}')
    print(f'Memory:  Number of valid graphs: {len(memory_valid)}')
    print(f'Memory:  Number of test graphs: {len(memory_test)}')


    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    mem_train_loader = DataLoader(memory_train, batch_size=64, shuffle=True)
    mem_test_loader = DataLoader(memory_test, batch_size=64, shuffle=True)
    for epoch in range(1000):
        Total, Gen, For =update_CL_(model, criterion, mem_train_loader,\
             train_loader, task=i)
        
        # Add the losses
        Total_loss.append(Total)
        Gen_loss.append(Gen)
        For_loss.append(For)
        
        # Add the accuracies
        test_acc = test(test_loader)
        mem_test_acc = test(mem_test_loader)
        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)
        
        # Print things when required
        if epoch%1==0:
            # scheduler.step()
            mem_train_acc = test(mem_train_loader)
            train_acc = test(train_loader)
            print("#########################################################################")
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'Mem Train Acc: {mem_train_acc:.4f}, Mem Test Acc: {mem_test_acc:.4f}')
            print("#########################################################################")

import matplotlib.pyplot as plt
plt.plot(accuracies_mem, label='memory accuracy')
plt.plot(accuracies_one, label='task accuracy')

# np.savetxt('withExtra_X_theta_Loss.csv', np.concatenate([ np.array(Total_loss).reshape([-1,1]),\
#                                          np.array(Gen_loss).reshape([-1,1]),\
#                                          np.array(For_loss).reshape([-1,1])], axis =1))
np.savetxt('withExtra_X_theta_Loss_higher.csv', np.concatenate([ np.array(Total_loss).reshape([-1,1]),\
                                         np.array(Gen_loss).reshape([-1,1]),\
                                         np.array(For_loss).reshape([-1,1])], axis =1))

np.savetxt('withExtra_X_theta_Acc_higher.csv', np.concatenate([ np.array(accuracies_one).reshape([-1,1]),\
                                         np.array(accuracies_mem).reshape([-1,1])], axis =1))


acc_old = np.loadtxt('blind.csv')
acc_mem = np.loadtxt('withMEM(EXP).csv')
acc_X = np.loadtxt('withExtra_X.csv')
acc_theta = np.loadtxt('withExtra_X_theta_Acc_higher.csv')

import matplotlib.pyplot as plt
import matplotlib
plt.figure(figsize=(50, 15))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)

# plt.plot(acc_old[:,0],label='memory accuracy (Naive)', lw = 2)
#plt.plot(acc_mem[:,0], label='task accuracy (Experience Replay)', lw = 2)
#plt.plot(acc_X[:,0], label='memory accuracy (J_x)', lw = 2)
plt.plot(acc_theta[:,0], label='task accuracy (Ours)', lw = 2)

#plt.plot(acc_old[:,1],'--', label='task accuracy (Naive)', lw = 3)
#plt.plot(acc_mem[:,1], '--', label='memory accuracy (Experience Replay)', lw = 3)
#plt.plot(acc_X[:,1], '--', label='task accuracy (J_x)', lw = 3)
plt.plot(acc_theta[:,1], '--', label='memory accuracy (Ours)', lw = 3)

plt.xlabel('Training Epoch k')
plt.ylabel('Test Accuracy')
plt.title('Task introduced at k = 200, 400')
plt.grid('True')
plt.legend()
plt.savefig('Accuracy_Plots.png', dpi=300)

Losses= np.loadtxt('withExtra_X_theta_Loss_higher.csv')


import matplotlib.pyplot as plt
import matplotlib
plt.figure(figsize=(30, 15))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

#plt.plot(acc_old[:,1],'--', label='task accuracy (Naive)', lw = 3)
plt.plot(Losses[:,2], label='Total Perturbations', lw = 3)
plt.plot(Losses[:,1], '.-', label='Impact of theta', lw = 3)
plt.plot(Losses[:,0], '--', label='Impact of x', lw = 3)
plt.legend()
plt.ylim([0,5])
plt.grid(True)
plt.savefig('Loss_PLot,png', dpi=300)


