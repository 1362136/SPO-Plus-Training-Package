from gurobipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import time

class SPOPlus:
    def __init__(self,X_train,optim_model):
        '''
        :param X_train: Data set that is used for training prediction model
        :param optim_model: A Gurobi Multi-Scenario Model
        '''
        self.__dict = {}
        self.__keys = {}
        self.X_train = X_train
        optim_model.update()
        self.optim_model = optim_model.copy()
        optim_model.optimize()
        optim_model.Params.ScenarioNumber = 0
        self.__dict[0] = self.optim_model.singleScenarioModel()
        self.c = torch.tensor(self.optim_model.singleScenarioModel().getAttr('Obj')).view(1, -1)
        self.__keys[self.c[0:1,:]] = self.optim_model.singleScenarioModel()
        for i in range(1, X_train.size()[0]):
            self.__dict[i] = self.optim_model.singleScenarioModel()
            self.c = torch.cat((self.c, torch.tensor(self.__dict[i].getAttr('Obj')).view(1, -1)),dim=0)
            self.__keys[self.c[i:i+1,:]] = self.optim_model.singleScenarioModel()
            self.optim_model.Params.ScenarioNumber += 1

    def get_output(self):
        '''
        :return: The objective coefficient vectors (tensor) which are the true outputs of the prediction problem
        '''
        return self.c

    def SPO_Plus_Loss(self, c, c_pred, reduction = 'mean'):
        '''
        :param c_pred: The outputs of the prediction model (tensor)
        :param reduction: Determines whether to return total or average loss. Takes values 'mean' or 'sum'.
        :return: the SPO+ loss value
        '''
        self.optim_model.update()
        dummy_model = self.optim_model.copy()
        dummy_model.setAttr('NumScenarios',c_pred.size()[0])
        dummy_model.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            dummy_model.setObjective(quicksum((-1*self.c[i:i+1,:] + 2*c_pred[i:i+1,:]).flatten().tolist()[k]
                                              * dummy_model.getVars()[k]
                                              for k in range(c_pred.size()[0])),GRB.MINIMIZE)
            dummy_model.params.ScenarioNumber += 1
        dummy_model.optimize()
        loss = 0
        dummy_model.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            loss += -1 * dummy_model.getAttr('ScenNObjVal') \
                    + 2 * torch.dot(c_pred[i:i+1,:].flatten(),torch.tensor(self.__keys[self.c[i:i+1,:]].getAttr('X'))) \
                    - self.__keys[self.c[i:i+1,:]].getAttr('ObjVal')
            dummy_model.params.ScenarioNumber += 1
        if reduction == 'sum':
            return loss
        elif reduction == 'mean':
            return loss / c_pred.size()[0]

    def SPO_loss(self, c, c_pred, reduction = 'mean'):
        '''
        :param c_pred: The outputs of the prediction model (tensor)
        :reduction: Determines whether to return total or average loss. Takes values 'mean' or 'sum'.
        :return: The SPO loss value
        '''
        self.dummy_model.setAttr('NumScenarios', c_pred.size()[0])
        self.dummy_model.Params.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            self.dummy_model.setAttr('ScenNObj',c_pred[i:i+1,:].flatten().numpy())
            self.dummy_model.params.ScenarioNumber += 1
        self.dummy_model.optimize()
        loss = 0
        self.dummy_model.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            loss += torch.dot(c[i:i+1,:].flatten(),self.dummy_model.getAttr('ScenNX')) \
                    - self.__keys[c[i:i+1,:]].getAttr('ObjVal')
        if reduction == 'sum':
            return loss
        elif reduction == 'mean':
            return loss / c_pred.size()[0]













