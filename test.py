from package import SPOPlus
import networkx as nx
import math
from gurobipy import *
import torch

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
    #print(list(grid.edges()))
    return grid

# First test is for the shortest path problem in which we have a grid DAG with weights given by c
X, c = generate_data(n=100, p=4, d=5, deg=3, epsilon=1)
grid = create_grid_graph(d = 5,c = c[0:1,:].flatten())
vertices = list(grid.nodes)
arcs = list(grid.edges)
cost = {arc: grid[arc[0]][arc[1]]['weight'] for arc in arcs}
#c_test = [float(i) for i in range(40)]

model = Model('multiscenario')
x = model.addVars(arcs,vtype=GRB.BINARY, name='x')
model.setObjective(quicksum(cost[arc] * x[arc] for arc in arcs), GRB.MINIMIZE)
model.addConstr(quicksum(x[(0,j)] for j in grid.successors(0)) == 1, name='source vertex constraint')
model.addConstrs(quicksum(x[(i,j)] for j in grid.successors(i)) -
                 quicksum(x[(j,i)] for j in grid.predecessors(i)) == 0 for i in vertices if int(i) != 0 and int(i)
                 != 25 - 1)
model.addConstr(quicksum(x[(j,25 - 1)] for j in grid.predecessors(25 - 1)) == 1)
model.NumScenarios = X.size()[0]
model.params.scenarioNumber = 0
for i in range(X.size()[0]):
    for arc in arcs:
        x[arc].setAttr('ScenNObj',c[i:i+1,:].flatten().tolist()[arcs.index(arc)])
    model.params.scenarioNumber += 1

#model.optimize()
#model.params.scenarioNumber = 0
#print(model.singleScenarioModel())
# model.params.scenarioNumber = 0
# for i in range(X.size()[0]):
#     #print(model.getAttr('ScenNObj'))
#     model.params.scenarioNumber += 1
# model.params.scenarioNumber = 2
# foo = model.singleScenarioModel()
# model.params.scenarioNumber = 3
# boo = model.singleScenarioModel()
# print(foo.getAttr('Obj'))
# print(boo.getAttr('Obj'))

#model.setObjective(quicksum(2 * x[arc] for arc in arcs), GRB.MINIMIZE)
#model.setObjective(quicksum(c_test[i] * model.getVars()[i] for i in range(len(c_test))), GRB.MINIMIZE)
#model.reset()
#model.setAttr('NumScenarios',5)
#print(len(l))
#print(model.getVars()[0] == x[(0,1)])
#print(x[(0,1)])
#print(model.getVars())
#print(model.getAttr('Obj'))
#model.reset()
#print(model.getAttr('NumScenarios'))
#model.NumScenarios =
#print(model.getObjective())
#model.optimize()
#model_2 = model.copy()
#model_2.reset()
#print(model_2.getVars())
#model_2.setObjective(quicksum(c_test[i] * model_2.getVars()[i] for i in range(40)),GRB.MINIMIZE)
#print(model_2.optimize())
#rint(model_2.getAttr('Obj'))
SPO_obj = SPOPlus.SPOPlus(X,model)
print(SPO_obj.SPO_Plus_Loss(c[1:2,:],torch.arange(20.,60.).view(1,40)))
print(SPO_obj.SPO_loss(c[1:2,:],torch.arange(20.,60.).view(1,40)))
