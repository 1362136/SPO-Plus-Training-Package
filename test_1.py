from package import SPOPlus
import networkx as nx
import math
from gurobipy import *
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def generate_data(n,p,d,deg,epsilon):
    '''
    :param n: Number of data points
    :param p: Dimension of each input vector
    :param d: dimension of the grid graph
    :param deg: degree of representing model misspecification
    :param epsilon: noise term
    :return:
    '''
    # Initialize the matrix of features; sample feature elements from N(0,1) dist.
    X = torch.randn(p,n)

    # Initialize matrix that maps features to cost vectors
    B = torch.bernoulli(0.5 * torch.ones(2*d*(d-1),p).float())

    # sample noise matrix from uniform dist. [1 - epsilon, 1 + epsilon]
    noise = 2.0 * epsilon * torch.rand(2*d*(d-1),n) + 1 - epsilon

    # Generate the output cost vectors from the input data
    c = (torch.pow(((1 / math.sqrt(p)) * torch.matmul(B,X) + 3.0), deg) + 1.0) * noise
    return torch.t(X), torch.t(c)

def create_grid_graph(d,c):
    '''
    :param d: denotes the dimensions of the grid graph
    :param c: denotes the weight vector for the graph
    :return: A n x n directed grid graph with weights from c.
    '''
    grid = nx.DiGraph()
    grid.add_edges_from([(j * d + i, j * d + i + 1) for j in range(d) for i in range(d-1)])
    grid.add_edges_from([(i + d * j, i + d * (j + 1)) for i in range(d) for j in range(d-1)])
    k = 0
    for e in list(grid.edges()):
        grid[e[0]][e[1]]['weight'] = float(c[k])
        k += 1
    return grid

# First test is for the shortest path problem in which we have a grid DAG with weights given by c


def sample_data_batch_SGD(size, n):
    sampled_index = torch.randint(high=n, size=(size,))
    return sampled_index

def train_batch_SPO_plus(SPO,model,opt,batch_size):
    sample_index = sample_data_batch_SGD(batch_size,1300)
    c_pred = torch.index_select(model(SPO.X_train),0,sample_index)
    c = torch.index_select(SPO.c,0,sample_index)
    l = SPO.SPO_Plus_Loss(c,c_pred)
    l.backward()
    opt.step()
    opt.zero_grad()

def train_batch(SPO,model,opt,batch_size,loss_func):
    sample_index = sample_data_batch_SGD(batch_size, 1300)
    c_pred = model(torch.index_select(SPO.X_train,dim = 0,index = sample_index))
    c = torch.index_select(SPO.c,dim = 0,index = sample_index)
    l = loss_func(c_pred,c)
    l.backward()
    opt.step()
    opt.zero_grad()


def fit_and_evaluate_SPO_plus(model,SPO,n_epochs,batch_size,optimizer,scheduler):
    # B = torch.zeros(n_epochs, 40, 4)  # The parameters of the model listed for each epoch.
    # running_avg_B = torch.zeros(n_epochs, 40, 4)
    # lr_list = []
    best_param = []
    best_loss = []
    best_loss.append(SPO.SPO_Plus_Loss(SPO.c_train, model(SPO.X_train)))
    best_param.append(list(model.parameters())[0].detach())
    for i in range(n_epochs):
        # for j in model.parameters():
        #     lr_list.append(scheduler.optimizer.param_groups[0]['lr'])
        #     B[i] = torch.tensor([lr_list[i]])*j.detach()
        # running_avg_B[i] = (1 / sum(lr_list)) * torch.sum(B, dim=0)
        sample_index = sample_data_batch_SGD(batch_size, 1300)
        c_pred = torch.index_select(model(SPO.X_train), 0, sample_index)
        c = torch.index_select(SPO.c, 0, sample_index)
        l = SPO.SPO_Plus_Loss(c, c_pred)
        l.backward()
        optimizer.step()
        x = SPO.SPO_Plus_Loss(SPO.c_train, model(SPO.X_train))
        if best_loss[-1] <= x:
            best_loss.append(best_loss[-1])
            best_param.append(best_param[-1])
        else:
            best_loss.append(x)
            best_param.append(list(model.parameters())[0].detach())
        optimizer.zero_grad()
        scheduler.step()
    model.weight = torch.nn.Parameter(best_param[-1])
    return SPO.SPO_loss(SPO.c_test, model(SPO.X_test), reduction = 'normalized')

def fit_and_evaluate(model,SPO,n_epochs,batch_size,optimizer,loss):
    for i in range(n_epochs):
        train_batch(SPO = SPO,model = model,opt = optimizer,batch_size = batch_size,loss_func = loss)
    return SPO.SPO_loss(SPO.c_test, model(SPO.X_test), reduction = 'normalized')





def SPO_experiment(degree):
    norm_loss_1 = []
    norm_loss_2 = []
    norm_loss_3 = []
    for i in range(3):
        X, c = generate_data(n=2000, p=4, d=5, deg=degree, epsilon=0.5)
        # Create the multi-scenario optimization model
        grid = create_grid_graph(d=5, c=c[0:1, :].flatten())
        vertices = list(grid.nodes)
        arcs = list(grid.edges)
        cost = {arc: grid[arc[0]][arc[1]]['weight'] for arc in arcs}
        model = Model('multiscenario')
        x = model.addVars(arcs, vtype=GRB.BINARY, name='x')
        model.setObjective(quicksum(cost[arc] * x[arc] for arc in arcs), GRB.MINIMIZE)
        model.addConstr(quicksum(x[(0, j)] for j in grid.successors(0)) == 1, name='source vertex constraint')
        model.addConstrs(quicksum(x[(i, j)] for j in grid.successors(i)) -
                         quicksum(x[(j, i)] for j in grid.predecessors(i)) == 0 for i in vertices if
                         int(i) != 0 and int(i)
                         != 25 - 1)
        model.addConstr(quicksum(x[(j, 25 - 1)] for j in grid.predecessors(25 - 1)) == 1)
        model.NumScenarios = X.size()[0]
        for j in range(X.size()[0]):
            for arc in arcs:
                x[arc].setAttr('ScenNObj', c[j:j + 1, :].flatten().tolist()[arcs.index(arc)])
            model.params.scenarioNumber += 1
        # Instantiate prediction models for experiment
        SPO_obj = SPOPlus.SPOPlus(X, model, train_test = 1300)
        model_1 = nn.Linear(4, 40, bias=False)
        model_2 = nn.Linear(4, 40, bias=False)
        model_3 = nn.Linear(4, 40, bias=False)
        opt_1 = optim.Adam(model_1.parameters(),lr=0.1)
        opt_2 = optim.Adam(model_2.parameters(),lr=0.1)
        opt_3 = optim.Adam(model_3.parameters(),lr=0.1)
        sched = optim.lr_scheduler.LambdaLR(opt_3,lambda epoch : (1 / (epoch + 1)))
        MSE = nn.MSELoss(reduction="mean")
        L1 = nn.L1Loss(reduction="mean")

        norm_loss_1.append(fit_and_evaluate(model_1, SPO_obj, 100, 10, opt_1, MSE))
        norm_loss_2.append(fit_and_evaluate(model_2, SPO_obj, 100, 10, opt_2, L1))
        norm_loss_3.append(fit_and_evaluate_SPO_plus(model_3, SPO_obj, 100, 10, opt_3, sched))
        x = len(norm_loss_1)
        return sum(norm_loss_1)/x, sum(norm_loss_2)/x, sum(norm_loss_3)/x

start = timer()
experiment_1 = SPO_experiment(degree = 1)
experiment_2 = SPO_experiment(degree = 2)
experiment_3 = SPO_experiment(degree = 4)
experiment_4 = SPO_experiment(degree = 6)
experiment_5 = SPO_experiment(degree = 8)
end = timer()
print((end - start) / 60)

degrees = [1,2,4,6,8]
MSE_losses = [experiment_1[0],experiment_2[0],experiment_3[0],experiment_4[0],experiment_5[0]]
L1_losses = [experiment_1[1],experiment_2[1],experiment_3[1],experiment_4[1],experiment_5[1]]
SPO_plus_losses = [experiment_1[2],experiment_2[2],experiment_3[2],experiment_4[2],experiment_5[2]]

x = np.arange(len(degrees))
width = 0.18
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, MSE_losses, width, label='MSE loss func')
rects2 = ax.bar(x, L1_losses, width, label='L1 loss func')
rects3 = ax.bar(x + width, SPO_plus_losses, width, label='SPO+ loss func')

ax.set_ylabel('Average Normalized SPO error')
ax.set_xlabel('Degree of Ground Truth')
ax.set_title('SPO+ vs MSE vs L1')
ax.set_xticks(x)
ax.set_xticklabels(degrees)
ax.legend()
plt.show()