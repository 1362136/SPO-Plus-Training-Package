from gurobipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys

class SPOPlus:
    def __init__(self,X_train,optim_model,optimizer,learning_model):
        '''
        :param X_train: Data set that is used for training prediction model
        :param optim_model: A Gurobi Multi-Scenario Model
        :param optimizer: Pytorch optimizer used for gradient descent
        :param learning_model: The prediction model, represented as a nn.Module object
        '''
        self.keys = {}
        self.X_train = X_train
        self.optim_model = optim_model
        self.learning_model = learning_model
        self.optimizer = optimizer
        self.optim_model.optimize()
        optim_model.ScenarioNumber = 0
        for i in range(X_train.size()[0]):
            self.keys[i] = (optim_model.fet_Attr('ScenNObj'),optim_model.getAttr('ScenNX'),optim_model.getAttr('ScenNObjVal'))
            optim_model.ScenarioNumber += 1


    def train_model(self):
        '''
        :return: The trained prediction model with SPO+ loss function
        '''

    def get_decisions(self, X):
        '''
        :param X: Test data that is to be fed into the model
        :return: The output of the prediction model or the coefficients of the obj function for the optim. problem
        '''





