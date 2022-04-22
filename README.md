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
5. Change optimization model function
```python
        def change_model(self, X_new, c_new, train_test):
        '''
        :param X_new: The new set of input data
        :param c_new: The new outputs
        :param train_test: The index where X_new is split into train and test data
        '''
```
### Workflow Example
