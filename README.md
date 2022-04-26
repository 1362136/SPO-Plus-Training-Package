# SPO-Plus-Training-Package

### Overview
This package faciliates training models that fall under the "predict then optimize" paradigm, in which the outputs of a supervised prediction model are leveraged as parameters for an underlying linear optimization problem. To that end, the implementation of a novel loss function - known as SPO-Plus loss function - used to train such models is provided by the package. Training "predict then optimize" models using the SPO-Plus loss function has the advantage of generating prediction models that minimize decision error rather prediction error. Decision error is induced when solving the optimization problem with a predicted parameter yields decisions that are sub-optimal. The package integrates with Gurobi API for python to solve linear optimization problems and takes full advantage of Pytorch's autograd feature.

### Functions
The package provides functions for evaluating the SPO-Plus and the SPO loss between predicted and true parameters. The SPO-Plus loss function is a convex surrogate of the SPO loss function; so the SPO-Plus loss function is used for training the prediction model and the SPO loss function is used as a metric of accuracy. In order to use the loss functions, an SPOPlus object must be created.

1. Constructor for generating an instance the SPOPlus class. Each input of the data set corresponds to a output label vector which is an objective coefficient vector for a scenario of the optimization model. The indices of  input vectors in the tensor X should be in numerical correspondence with the scenarios of the multisceanrio model. Also, each scenario should have the same constraints and decision variables; the objective coefficients are the only parametrs that change across scenarios.
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
### Workflow Example
We go through the pipeline of setting up the Gurobi Multi-Scenario model for the Shortest Paths LP on a 5 x 5 grid graph. The edge costs of the graph can be represented as 40 dimensional vector. We will have 1000 such vectors aggregated into a 1000 x 40 dimensional tensor called c. Each of the vectors in c will be associated with some input vector of dimension 4. Thus we will have a 1000 such vectors aggregated into a 1000 x 4 dimensional tensor called X.  First, we synthetically generate X and c.
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
After we generate the data, we create the 5 x 5 grid graph using the tensor c.
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
