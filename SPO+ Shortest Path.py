import networkx as nx
from gurobipy import *
import torch


def create_grid_graph(n,c):
    '''
    :param n: denotes the dimensions of the grid graph
    :param c: denotes the weight vector for the graph
    :return: A n x n directed grid graph with weights from c.
    '''
    grid = nx.DiGraph()
    grid.add_edges_from([(j * n + i, j * n + i + 1) for j in range(n-1) for i in range(n-1)])
    grid.add_edges_from([(i + n * j, i + n * (j + 1)) for i in range(n-1) for j in range(n-1)])
    i = 0
    for e in list(grid.edges()):
        grid[e[0]][e[1]]['weight'] = c[i]
        i += 1
    return grid

def generate_data(n,p,deg,epsilon):
    # Initialize the matrix of features; sample feature elements from N(0,1) dist.
    X = torch.randn(p,n)

    # Initialize matrix that maps features to cost vectors
    B = torch.bernoulli(0.5 * torch.ones(2 * n * (n - 1),p).float())

    # sample noise matrix from uniform dist. [1 - epsilon, 1 + epsilon]
    noise = 2.0 * epsilon * torch.rand(n, 2 * n * (n - 1)) + 1 - epsilon

    # Generate the output cost vectors from the input data
    c = (torch.pow(((1 / torch.sqrt(p)) * torch.matmul(B,X) + 3.0), deg) + 1.0) * noise

    return X, c

def shortest_paths_oracle_1(n, c):
    '''
    Computes the optimal solution to the shortest paths Linear program using dynamic programming linear time algorithm.
    :param c: The weight vector that is used so solve the shortest paths problem.
    :param n: Dimensions of grid graph.
    '''
    grid = create_grid_graph(n,c)

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
    cost = {arc: grid[arc[0]][arc[1]]['weight'] for arc in arcs}

    x = model.addVars(arcs,vtype=GRB.BINARY, name='x')
    model.setObjective(quicksum(cost[arc] * x[arc] for arc in arcs), GRB.MINIMIZE)
    model.addConstr(quicksum(x[(0,j)] for j in grid.successors(0)) == 1, name='source vertex constraint')
    model.addConstrs(quicksum(x[(i,j)] for j in grid.successors(i)) -
                     quicksum(x[(j,i)] for j in grid.predecessors(i)) for i in vertices if int(i) != 0 and int(i) != n*n)
    model.addConstr(quicksum(x[(j,n*n)] for j in grid.predecessors(n*n)) == -1)

shortest_paths_oracle_2(2, [1,2,3,4])