import networkx as nx
from gurobipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


def create_grid_graph(n,c):
    '''
    :param n: denotes the dimensions of the grid graph
    :param c: denotes the weight vector for the graph
    :return: A n x n directed grid graph with weights from c.
    '''
    grid = nx.DiGraph()
    grid.add_edges_from([(j * n + i, j * n + i + 1) for j in range(n) for i in range(n-1)])
    grid.add_edges_from([(i + n * j, i + n * (j + 1)) for i in range(n) for j in range(n-1)])
    i = 0
    for e in list(grid.edges()):
        grid[e[0]][e[1]]['weight'] = c[i]
        i += 1
    return grid


def generate_data(n,p,deg,epsilon):
    '''
    :param n: Number of data points
    :param p: Dimension of each input vector
    :param deg: degree of representing model misspecification
    :param epsilon: noise term
    :return:
    '''
    # Initialize the matrix of features; sample feature elements from N(0,1) dist.
    X = torch.randn(p,n)

    # Initialize matrix that maps features to cost vectors
    B = torch.bernoulli(0.5 * torch.ones(2 * n * (n - 1),p).float())

    # sample noise matrix from uniform dist. [1 - epsilon, 1 + epsilon]
    noise = 2.0 * epsilon * torch.rand(2 * n * (n - 1),n) + 1 - epsilon

    # Generate the output cost vectors from the input data
    c = (torch.pow(((1 / math.sqrt(p)) * torch.matmul(B,X) + 3.0), deg) + 1.0) * noise

    return X, c


def shortest_paths_oracle_1(n, c):
    '''
    Computes the optimal solution to the shortest paths Linear program using dynamic programming linear time algorithm.
    :param c: The weight vector that is used so solve the shortest paths problem.
    :param n: Dimensions of grid graph.
    '''
    grid = create_grid_graph(n,c)
    infinity = np.inf


    return


def shortest_paths_oracle_2(n, c):
    '''
    Computes the optimal solution to the shortest paths Linear Program using Gurobi Optimizer
    :param c: The weight vector that is used so solve the shortest paths problem.
    :param n: Dimensions of grid graph.
    :return: minimizer of the shortest paths problem
    '''
    model = Model('Shortest Path')

    grid = create_grid_graph(n, c)
    vertices = list(grid.nodes)
    arcs = list(grid.edges)
    #print(arcs)
    cost = {arc: grid[arc[0]][arc[1]]['weight'] for arc in arcs}
    x = model.addVars(arcs,vtype=GRB.BINARY, name='x')
    model.setObjective(quicksum(cost[arc] * x[arc] for arc in arcs), GRB.MINIMIZE)
    model.addConstr(quicksum(x[(0,j)] for j in grid.successors(0)) == 1, name='source vertex constraint')
    model.addConstrs(quicksum(x[(i,j)] for j in grid.successors(i)) -
                     quicksum(x[(j,i)] for j in grid.predecessors(i)) == 0 for i in vertices if int(i) != 0 and int(i)
                     != n*n - 1)
    model.addConstr(quicksum(x[(j,n*n - 1)] for j in grid.predecessors(n*n - 1)) == 1)
    model.optimize()
    return torch.tensor([y for y in model.getAttr('X')]), model.ObjVal


def SPO_plus_loss_function(c_pred, c):
    loss = 0
    for i in range(c.size()[1]):
        loss += (-1 * shortest_paths_oracle_2(5, (-c[:,i:i+1].flatten() + 2 * c_pred[:,i:i+1]).flatten())[1]
                 + 2 * torch.dot(c_pred[:,i:i+1].flatten(),
                 shortest_paths_oracle_2(5, c[:,i:i+1].flatten())[0]) -
                 shortest_paths_oracle_2(5, c[:,i:i+1].flatten())[1])
    return (1/c.size()[1]) * loss

# Testing if torch autograd will automatically compute gradients of SPO+ loss.

W = torch.zeros(40,4,requires_grad=True)
learning_rate = 0.1
X, output = generate_data(n=100, p=4, deg=3, epsilon=0.1)
model = nn.Linear(4,40)
errors = []
optimizer = optim.Adam(model.parameters())
#ef forward(x):
    #return torch.matmul(W,x)


for i in range(40):
    pred = model(torch.t(X))
    l = SPO_plus_loss_function(pred,output)
    l.backward()
    with torch.no_grad():
        optimizer.step()
    W.grad.zero_()
    if i % 2 == 0:
        errors.append(l.detach())

print(errors)