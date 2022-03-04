import networkx as nx
from gurobipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt


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
    print(list(grid.edges()))
    return grid


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
    return X, c


def shortest_paths_oracle_1(d, c):
    '''
    Computes the optimal solution to the shortest paths Linear program using dynamic programming linear time algorithm.
    :param c: The weight vector that is used so solve the shortest paths problem.
    :param d: Dimensions of grid graph.
    '''
    grid = create_grid_graph(d,c)
    infinity = np.inf
    return


def shortest_paths_oracle_2(d, c):
    '''
    Computes the optimal solution to the shortest paths Linear Program using Gurobi Optimizer
    :param c: The weight vector that is used so solve the shortest paths problem.
    :param d: Dimensions of grid graph.
    :return: minimizer of the shortest paths problem
    '''
    model = Model('Shortest Path')

    grid = create_grid_graph(d, c)
    vertices = list(grid.nodes)
    arcs = list(grid.edges)
    #print(arcs)
    cost = {arc: grid[arc[0]][arc[1]]['weight'] for arc in arcs}
    x = model.addVars(arcs,vtype=GRB.BINARY, name='x')
    model.setObjective(quicksum(cost[arc] * x[arc] for arc in arcs), GRB.MINIMIZE)
    model.addConstr(quicksum(x[(0,j)] for j in grid.successors(0)) == 1, name='source vertex constraint')
    model.addConstrs(quicksum(x[(i,j)] for j in grid.successors(i)) -
                     quicksum(x[(j,i)] for j in grid.predecessors(i)) == 0 for i in vertices if int(i) != 0 and int(i)
                     != d*d - 1)
    model.addConstr(quicksum(x[(j,d*d - 1)] for j in grid.predecessors(d*d - 1)) == 1)
    model.optimize()
    return c.flatten(), torch.tensor([y for y in model.getAttr('X')]), float(model.ObjVal)


def sample_data_batch_SGD(size, n):
    sampled_index = torch.randint(high=n, size=(5,))
    return sampled_index


def SPO_plus_loss_function_epoch(c_pred, M):
    loss = 0
    for i in range(c_pred.size()[1]):
        loss += (-1 * shortest_paths_oracle_2(5, -1*M[i][0] + 2 * c_pred[:,i:i+1].flatten())[2]
                 + 2 * torch.dot(c_pred[:,i:i+1].flatten(), M[i][1])
                 - M[i][2])
    return (1/c_pred.size()[1]) * loss


def loss_function(model,loss_func,X,c):
    l = loss_func(model(torch.t(X)),torch.t(c)).item()
    return l


def SPO_plus_loss(c_pred, M, sample_index):
    loss = 0
    for i in sample_index:
        loss += (-1 * shortest_paths_oracle_2(5, (-1 * M[i][0] + 2 * c_pred[:,i:i+1].flatten()))[2]
                 + 2 * torch.dot(c_pred[:, i:i+1].flatten(), M[i][1].flatten())
                 - M[i][2])
    return (1 / sample_index.size()[0]) * loss


def train_batch_SPO_plus(X,M,model,opt,batch_size):
    sample_index = sample_data_batch_SGD(batch_size,100)
    pred = torch.t(model(torch.t(X)))
    l = SPO_plus_loss(pred, M, sample_index)
    l.backward()
    opt.step()
    opt.zero_grad()

def train_batch(X,c,model,opt,batch_size,loss_func):
    sample_index = sample_data_batch_SGD(batch_size, 100)
    pred = model(torch.t(torch.index_select(X,dim = 1,index = sample_index)))
    c_batch = torch.t(torch.index_select(c,dim = 1,index = sample_index))
    l = loss_func(pred,c_batch)
    l.backward()
    opt.step()
    opt.zero_grad()


def fit_and_evaluate_SPO_plus(model,X,output,n_epochs,batch_size,optimizer):
    errors = []
    epoch = []
    M = []  # List used to cache optimization information for each element in the output. Eases computation time
    B = torch.zeros(n_epochs, 40, 4)  # The parameters of the model listed for each epoch.
    running_avg_B = torch.zeros(n_epochs, 40, 4)
    for i in range(output.size()[1]):
        M.append(shortest_paths_oracle_2(5, output[:, i:i + 1].flatten()))
    for i in range(n_epochs):
        for j in model.parameters():
            B[i] = j.detach()
        running_avg_B[i] = (1 / (i + 1)) * torch.sum(B, dim=0)
        #with torch.no_grad():
            #epoch.append(i)
            #pred = torch.t(model(torch.t(X)))
            #errors.append(SPO_plus_loss_function_epoch(pred, M))
        train_batch_SPO_plus(X, M, model, optimizer, batch_size)
        #for i in model.parameters():
            #i = running_avg_B[n_epochs-1]
        model.weight = torch.nn.Parameter(running_avg_B[n_epochs-1])
    return normalized_SPO_loss(X,output,model)

def fit_and_evaluate(model,X,output,n_epochs,batch_size,optimizer,loss):
    errors = []
    epoch = []
    for i in range(n_epochs):
        #with torch.no_grad():
            #epoch.append(i)
            #errors.append(loss_function(model = model,loss_func = loss,X = X,c = output))
        train_batch(X = X,c = output,model = model,opt = optimizer,batch_size = batch_size,loss_func = loss)
    return normalized_SPO_loss(X,output,model)

def normalized_SPO_loss(X,c,model):
    M = []
    for i in range(c.size()[1]):
        M.append(shortest_paths_oracle_2(5, c[:,i:i+1].flatten())[2])
    loss = 0
    for i in range(c.size()[1]):
        loss += torch.dot(shortest_paths_oracle_2(5, model(torch.t(X))[i:i+1,:].flatten())[1],c[:,i:i+1].flatten()) \
                - M[i]
    return loss / sum(M)


def SPO_experiment(degree):
    norm_loss_1 = []
    norm_loss_2 = []
    norm_loss_3 = []
    for i in range(50):
        learning_rate = 0.1
        X, output = generate_data(n=1000, p=4, d=5, deg=degree, epsilon=1)
        model_1 = nn.Linear(4, 40, bias=False)
        model_2 = nn.Linear(4, 40, bias=False)
        model_3 = nn.Linear(4, 40, bias=False)
        opt_1 = optim.Adam(model_1.parameters())
        opt_2 = optim.Adam(model_2.parameters())
        opt_3 = optim.Adam(model_3.parameters())
        MSE = nn.MSELoss(reduction="mean")
        L1 = nn.L1Loss(reduction="mean")
        # weight_matrix  = fit_and_evaluate_SPO_plus(model_1,X,output,500,5,opt)[2]
        norm_loss_1.append(fit_and_evaluate(model_1, X, output, 500, 5, opt_1, MSE))
        norm_loss_2.append(fit_and_evaluate(model_2, X, output, 500, 5, opt_2, L1))
        norm_loss_3.append(fit_and_evaluate_SPO_plus(model_3, X, output, 500, 5, opt_3))
        x = len(norm_loss_1)
        return sum(norm_loss_1)/x, sum(norm_loss_2)/x, sum(norm_loss_3)/x


experiment_1 = SPO_experiment(degree = 1)
experiment_2 = SPO_experiment(degree = 2)
experiment_3 = SPO_experiment(degree = 4)
experiment_4 = SPO_experiment(degree = 6)
experiment_5 = SPO_experiment(degree = 8)

degrees = [1,2,4,6,8]
MSE_losses = [experiment_1[0],experiment_2[0],experiment_3[0],experiment_4[0],experiment_5[0]]
L1_losses = [experiment_1[1],experiment_2[1],experiment_3[1],experiment_4[1],experiment_4[1]]
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