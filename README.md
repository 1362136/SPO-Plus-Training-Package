# SPO-Plus-Training-Package

### Overview
This package faciliates training models that fall under the "predict then optimize" paradigm, in which the outputs of a supervised prediction model are leveraged as parameters for an underlying linear optimization problem. To that end, the implementation of a novel loss function - known as SPO-Plus loss function - used to train such models is provided by the package. Training "predict then optimize" models using the SPO-Plus loss function has the advantage of generating prediction models that minimize decision error rather prediction error. Decision error is induced when solving the optimization problem with a predicted parameter yields decisions that are sub-optimal. The package integrates with Gurobi API for python to solve linear optimization problems and takes full advantage of Pytorch's autograd feature.

### Functions
The package provides functions for evaluating the SPO-Plus and the SPO loss between predicted and true parameters. The SPO-Plus loss function is a convex surrogate of the SPO loss function; so the SPO-Plus loss function is used for training the prediction model and the SPO loss function is used as a metric of accuracy. In order to use the loss functions, an SPOPlus object must be created.

1. Constructor for generating an instance the SPOPlus class. Each input of the data set corresponds to a output label vector which is an objective coefficient vector for a scenario of the optimization model. The indices of  input vectors in the tensor X should be in numerical correspondence with the scenarios of the multisceanrio model. Also, each scenario should have the same constraints and decision variables; the objective coefficients are the only parameters that change across scenarios.
```python
def __init__(self,X,optim_model,train_test):
'''
:param X: Input data set that is used for the prediction model (tensor)
:param optim_model: A Gurobi Multi-Scenario Model
'''  
```
2. Function used to obtain true outputs of the prediction problem
```python
def get_output(self):
'''
:return: The objective coefficient vectors (tensor) which are the true outputs of the prediction model
'''

```
3. SPO_Plus loss function
```python
 def SPO_Plus_Loss(self, c, c_pred, reduction = 'mean'):
 '''
 :param c: The true ouputs of the prediction model (tensor)
 :param c_pred: The predicted outputs of the prediction model (tensor)
 :param reduction: Determines whether to return total or average loss. Takes values 'mean' or 'sum'.
 :return: the SPO+ loss value
 '''
```
4. SPO loss function
```python
    def SPO_loss(self, c, c_pred, reduction = 'mean'):
    '''
    :param c_pred: The outputs of the prediction model (tensor)
    :reduction: Determines whether to return total, average, or normalized SPO loss. Takes values 'mean','sum', or 'normalized'
    :return: The SPO loss value
    '''
```
5. Change optimization model function. Changes the input and ouput data of the prediction model while leaving the constraints and decision variables of the optimization problem the same.
```python
        def change_model(self, X_new, c_new, train_test):
        '''
        :param X_new: The new set of input data
        :param c_new: The new outputs
        :param train_test: The index where X_new is split into train and test data
        '''
```
### SPOPlus Object Attributes
After creating an instance of the SPOPlus class using the constructor, one can access the attributes `X_test`, `X_train`, `c_test`, and `c_train` to faciliate training and evaluating the prediction models.
### Workflow Example
We go through the pipeline of setting up the Gurobi Multi-Scenario model for the Shortest Paths LP on a 5 x 5 grid graph. The edge costs of the graph can be represented as 40 dimensional vector. We will have 1000 such vectors aggregated into a 1000 x 40 dimensional tensor called c. Each of the vectors in c will be associated with some input vector of dimension 4. Thus we will have a 1000 such vectors aggregated into a 1000 x 4 dimensional tensor called X.  First, we synthetically generate X and c. After we generate the data, we create the 5 x 5 grid graph using the NetworkX library and using the tensor c. Finally we create the Gurobi Multiscenario model (each scenario will have a different vector in c as the objective coeffcients) for the shortest paths linear program. The code below shows the functions that are used to generate the data and create the grid graph.
```python
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
```
```python
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
```
In the case we are working in, we make the following calls to generate the data
```python
        X, c = generate_data(n=1000, p=4, d=5, deg=degree, epsilon=0.5)
        grid = create_grid_graph(d=5, c=c[0:1, :].flatten())
```
Note that we are setting the initial weights of the graph as the first vector in the tensor c; however, this choice of initial weights does not matter as the only purpose for creating graph is to faciliate setting up the Gurobi Optimization problem.
```python
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
```
The first two lines of the code create index lists of the nodes and edges of the graph. Starting from line 4, we initialize the Gurobi model and add decision variables and constraints that are consistent with the Linear program formulation for the shortest paths problem and iterate through the number of scenarios in the model to change the objective coefficients. We then feed the Gurobi Model and the tensor X into the constructor of the SPOPlus class.
```python
from gurobipy import*
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.rand(1000,10)
c = torch.rand(1000,5)
pred_model = nn.Linear(10,5)
optimizer = optim.SGD(pred_model.parameters(),lr = 0.1)

SPO = SPOPlus.SPOPlus(X, model, train_test=700)
```
Creating the constructor automatically splits the data into training and test sets. We can retrieve test data by the object attributes `X_test`, `X_train`, `c_test`, `c_train`. Once the constructor is created and once a prediction model is defined, one can use pytorch's autograd functionality following to compute subgradients of the loss function
```python
for i in range(100):
    loss  = SPO.SPO_Plus_Loss(c, pred_model(X))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
```
