import numpy as np
import torch
from torch import nn
from package import SPOPlus
import pandas as pd
import copy
import random
from torch import optim
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from gurobipy import *
from sklearn.preprocessing import MinMaxScaler


cov_matrix = [[0.06361574, 0.00757464, 0.00448418],
       [0.00757464, 0.06911954, 0.00376472],
       [0.00448418, 0.00376472, 0.02332345]]

cov = {(i,j) : cov_matrix[i][j] for i in range(3) for j in range(3)}
# X = torch.tensor(pd.read_csv('inputs.csv').drop(['Date'],axis = 1).to_numpy()).float()
y = np.delete(pd.read_csv('outputs.csv').to_numpy(),0,axis = 1)
# scaler = MinMaxScaler(feature_range = (-1,1))
# y = scaler.fit_transform(y)
# X = X[1:,:]
y = y[1:,:]


def split_data(stock, lookback):
    data_raw = stock  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    train_set_size = 700

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return x_train, y_train, x_test, y_test


X_train, y_train, X_test, y_test = split_data(y, 56)

X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

class LSTM(nn.Module):
    def __init__(self,batch_size):
        super(LSTM, self).__init__()
        self.RNN = nn.LSTM(input_size=3,hidden_size=16,batch_first=True)
        self.linear = nn.Linear(16,3)
        self.batch_Size = batch_size
    def forward(self,x):
        h0 = torch.randn(1, self.batch_Size, 16)
        c0 = torch.randn(1, self.batch_Size, 16)
        x = self.RNN(x,(h0,c0))[0]
        return self.linear(x[:,-1,:])

def train_batch_SPO_plus(X_train,c_train,SPO,model,opt):
    sample_index = torch.from_numpy(np.random.randint(X_train.size()[0],size = 20)).to(torch.long)
    x_train = torch.index_select(X_train,0,sample_index)
    c = torch.index_select(c_train,0,sample_index)
    l = SPO.SPO_Plus_Loss(c,model(x_train))
    l.backward()
    opt.step()
    opt.zero_grad()

def train_batch(X_train,c_train,SPO,model,opt,loss_func):
    sample_index = torch.from_numpy(np.random.randint(X_train.size()[0],size = 20)).to(torch.long)
    x_train = torch.index_select(X_train,0,sample_index)
    c = torch.index_select(c_train,dim = 0,index = sample_index)
    l = loss_func(c,model(x_train))
    l.backward()
    opt.step()
    opt.zero_grad()

def fit_and_evaluate_SPO_plus(X_train,c_train,X_test,c_test,model,SPO,n_epochs,batch_size,optimizer,scheduler):
    lr_list = []
    for i in range(n_epochs):
        for j in model.parameters():
            lr_list.append(scheduler.optimizer.param_groups[0]['lr'])
        train_batch_SPO_plus(X_train,c_train,SPO,model,optimizer)
        scheduler.step()
    model.batch_Size = X_test.size()[0]
    return SPO.SPO_loss(c_test, model(X_test), reduction = 'normalized')

def fit_and_evaluate(X_train,c_train,X_test,c_test,model,SPO,n_epochs,batch_size,optimizer,loss,scheduler):
    for i in range(n_epochs):
        train_batch(X_train=X_train,c_train=c_train,SPO = SPO,model = model,opt = optimizer,loss_func = loss)
        scheduler.step()
    model.batch_Size = X_test.size()[0]
    return SPO.SPO_loss(c_test, model(X_test), reduction = 'normalized')

def cross_validation_SPO(grid,num_folds,SPO,n_epochs,batch_size):
    average_scores = []
    X_folds = list(torch.split(tensor=SPO.X_train,split_size_or_sections=SPO.X_train.size()[0] // num_folds))
    c_folds = list(torch.split(tensor=SPO.c_train,split_size_or_sections=SPO.c_train.size()[0] // num_folds))
    idx_list = torch.arange(len(X_folds)).tolist()
    for i in grid:
        sum = 0
        for j in tqdm(range(len(X_folds))):
            model = LSTM(batch_size=20)
            opt = optim.Adam(model.parameters(), lr=0.1,weight_decay=i)
            sched = optim.lr_scheduler.LambdaLR(opt, lambda epoch: (1 / (math.sqrt(epoch + 1))))
            valid = X_folds[j]
            idx_list.remove(j)
            train = X_folds[idx_list[0]]
            out_train = c_folds[idx_list[0]]
            for k in range(1, len(idx_list)):
                train = torch.cat((train, X_folds[idx_list[k]]), dim=0)
                out_train = torch.cat((out_train, c_folds[idx_list[k]]), dim=0)
            sum += fit_and_evaluate_SPO_plus(train,out_train,valid,c_folds[j],model,SPO,n_epochs,batch_size,opt,sched)
            idx_list.insert(j, j)
        average_scores.append(sum / len(X_folds))
    opt_param = grid[average_scores.index(min(average_scores))]
    model = LSTM(batch_size=20)
    opt = optim.Adam(model.parameters(), lr=0.1, weight_decay=opt_param)
    sched = optim.lr_scheduler.LambdaLR(opt, lambda epoch: (1 / (math.sqrt(epoch + 1))))
    return fit_and_evaluate_SPO_plus(SPO.X_train,SPO.c_train,SPO.X_test,SPO.c_test,model,SPO,n_epochs,batch_size,opt,sched)

def cross_validation(grid,num_folds,SPO,n_epochs,batch_size,loss):
    average_scores = []
    X_folds = list(torch.split(tensor=SPO.X_train,split_size_or_sections=SPO.X_train.size()[0] // num_folds))
    c_folds = list(torch.split(tensor=SPO.c_train,split_size_or_sections=SPO.c_train.size()[0] // num_folds))
    idx_list = torch.arange(len(X_folds)).tolist()
    for i in grid:
        sum = 0
        for j in tqdm(range(len(X_folds))):
            model = LSTM(batch_size=20)
            opt = optim.Adam(model.parameters(), lr=0.05,weight_decay=i)
            sched = optim.lr_scheduler.LambdaLR(opt, lambda epoch: (1 / (math.sqrt(epoch + 1))))
            valid = X_folds[j]
            idx_list.remove(j)
            train = X_folds[idx_list[0]]
            out_train = c_folds[idx_list[0]]
            for k in range(1,len(idx_list)):
                train = torch.cat((train,X_folds[idx_list[k]]),dim=0)
                out_train = torch.cat((out_train, c_folds[idx_list[k]]), dim=0)
            sum += fit_and_evaluate(train,out_train,valid,c_folds[j],model,SPO,n_epochs,batch_size,opt,loss,sched)
            idx_list.insert(j,j)
        average_scores.append(sum / len(X_folds))
    opt_param = grid[average_scores.index(min(average_scores))]
    model = LSTM(batch_size=20)
    opt = optim.Adam(model.parameters(), lr=0.05, weight_decay=opt_param)
    sched = optim.lr_scheduler.LambdaLR(opt, lambda epoch: (1 / (math.sqrt(epoch + 1))))
    return fit_and_evaluate(SPO.X_train,SPO.c_train,SPO.X_test,SPO.c_test,model,SPO,n_epochs,batch_size,opt,loss,sched)


y = torch.cat((y_train,y_test),0)
X = torch.cat((X_train,X_test),0)
cost = y[0:1,:].flatten().tolist()
model = Model('Portfolio_Optimization')
indices_1 = [i for i in range(3)]
indices_2 = [i for i in range(3)]
x = model.addVars(indices_1,lb = 0,name = 'x')
model.setObjective(quicksum(-1*cost[i]*x[i] for i in indices_1), GRB.MINIMIZE)
model.addConstr(quicksum(x[i] for i in indices_1) == 1)
model.addConstr(quicksum(cov[i,j]*x[i]*x[j] for i in indices_1 for j in indices_2) <= 0.05)
model.numScenarios = X.size()[0]
for j in range(X.size()[0]):
    for i in indices_1:
        x[i].setAttr('ScenNObj', y[j:j + 1, :].flatten().tolist()[i])
    model.params.scenarioNumber += 1

SPO = SPOPlus.SPOPlus(X, model, train_test=700)
MSE = nn.MSELoss(reduction="mean")
L1 = nn.L1Loss(reduction="mean")
grid = np.logspace(-6,2,num=10)

list_1 = []
list_2 = []
list_3 = []

for i in tqdm(range(10)):
    list_1.append(cross_validation(grid, 4, SPO, 100, 20, MSE))
    list_2.append(cross_validation(grid, 4, SPO, 100, 20, L1))
    list_3.append(cross_validation_SPO(grid, 4, SPO, 100, 20))

list_1 = [abs(x) for x in list_1]
list_2 = [abs(x) for x in list_2]
list_3 = [abs(x) for x in list_3]
# fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
# ax1.boxplot(list_1)
# ax2.boxplot(list_2)
# ax3.boxplot(list_3)
# plt.show()


fig = plt.figure(figsize = (10, 5))
loss_func = ['MSE', 'L1', 'SPO+']
Average_SPO_error = [sum(list_1) / 5, sum(list_2) / 5, sum(list_3) / 5]
plt.bar(loss_func,Average_SPO_error,width = 0.4)
plt.xlabel("Loss Functions")
plt.ylabel("Out of Sample SPO loss")
plt.show()
