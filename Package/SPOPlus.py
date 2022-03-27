from gurobipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys

class SPOPlus:
    def __init__(self,X_train,optim_model):
        '''
        :param X_train: Data set that is used for training prediction model
        :param optim_model: A Gurobi Multi-Scenario Model
        '''
        self.__dict = {}
        self.__keys = {}
        self.X_train = X_train
        self.optim_model = optim_model
        self.dummy_model = optim_model.copy()
        optim_model.ScenarioNumber = 0
        optim_model.optimize()
        self.__dict[0] = self.optim_model.SingleScenarioModel()
        self.c = torch.tensor(self.optim_model.SingleScenarioModel().getAttr('Obj')).view(1, -1)
        self.__keys[self.c] = 0
        for i in range(1, X_train.size()[0]):
            self.__dict[i] = self.optim_model.SingleScenarioModel()
            self.c = torch.cat(self.c, torch.tensor(self.__dict[i].getAttr('Obj')).view(1, -1))
            self.__keys[self.c[i:i+1,:]] = self.optim_model.SingleScenarioModel()
            self.optim_model.ScenarioNumber += 1

    def get_output(self):
        '''
        :return: The objective coefficient vectors (tensor) which are the true outputs of the prediction problem
        '''
        return self.c

    def SPO_Plus_Loss(self,c_pred, reduction):
        '''
        :param c_pred: The outputs of the prediction model (tensor)
        :return: the loss value
        '''
        self.dummy_model.setAttr('NumScenarios',c_pred.size()[0])
        self.dummy_model.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            self.dummy_model.setAttr('ScenNObj',(-self.c[i:i+1,:] + 2*c_pred[i:i+1,:]).flatten().numpy())
            self.dummy_model.ScenarioNumber += 1
        self.dummy_model.optimize()
        loss = 0
        self.dummy_model.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            loss += -1 * self.dummy_model.getAttr('ScenNObjVal') \
                    + 2 * torch.dot(c_pred[i:i+1,:].flatten(),torch.tensor(self.__keys[self.c[i:i+1,:]].getAttr('X'))) \
                    - self.__keys[self.c[i:i+1,:]].getAttr('ObjVal')
        if reduction == 'sum':
            return loss
        elif reduction == 'mean':
            return loss / c_pred.size()[0]











