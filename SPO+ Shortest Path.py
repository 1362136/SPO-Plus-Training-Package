import networkx as nx
from gurobipy import*
import torch

def create_grid_graph(n):
    grid = nx.DiGraph()
    grid.add_edges_from([(j * n + i, j * n + i + 1) for j in range(n) for i in range(n)])
    grid.add_edges_from([(i + n * j, i + n * (j + 1)) for i in range(n) for j in range(n)])
    return grid

def generate_data(n,p,d,deg,epsilon):
    # Initialize the matrix of features; sample feature elements from N(0,1) dist.
    X = torch.randn(p,n)

    # Initialize matrix that maps features to cost vectors
    B = torch.bernoulli(0.5 * torch.ones(d,p).float())

    # sample noise matrix from uniform dist. [1 - epsilon, 1 + epsilon]
    noise = 2.0 * epsilon * torch.rand(n,d) + 1 - epsilon

    # Generate the output cost vectors from the input data
    c = (torch.pow(((1 / torch.sqrt(p)) * torch.matmul(B,X) + 3.0), deg) + 1.0) * noise

    return X, c