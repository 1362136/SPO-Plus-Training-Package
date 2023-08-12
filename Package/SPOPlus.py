from gurobipy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import time

class SPOPlus:
    def __init__(self,X,optim_model,train_test):
        '''
        :param X: Data set that is used for training prediction model
        :param optim_model: A Gurobi Multi-Scenario Model
        :param train_test: The index where X is split into train and test data
        '''
        self.X = X
        self.X_train = X[0:train_test]
        self.X_test = X[train_test:]
        self.__x = {}
        self.__obj_val = {}
        optim_model.update()
        self.optim_model = optim_model.copy()
        self.optim_model.optimize()
        self.optim_model.Params.ScenarioNumber = 0
        self.__x[0] = self.optim_model.getAttr('ScenNX')
        self.__obj_val[0] = self.optim_model.getAttr('ScenNObjVal')
        self.c = torch.tensor(self.optim_model.getAttr('ScenNObj')).view(1, -1)
        # self.__keys[0] = self.optim_model.singleScenarioModel()
        for i in range(1, X.size()[0]):
            self.__x[i] = self.optim_model.getAttr('ScenNX')
            self.__obj_val[i] = self.optim_model.getAttr('ScenNObjVal')
            self.c = torch.cat((self.c, torch.tensor(self.optim_model.getAttr('ScenNObj')).view(1, -1)),dim=0)
            # self.__keys[i] = self.optim_model.singleScenarioModel()
            self.optim_model.Params.ScenarioNumber += 1
        self.c_train = self.c[0:train_test]
        self.c_test = self.c[train_test:]

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
            for j in range(c_pred.size()[1]):
                dummy_model.getVars()[j].setAttr('ScenNObj',(-1*c[i:i+1,:] + 2*c_pred[i:i+1,:]).flatten().tolist()[j])
            dummy_model.params.ScenarioNumber += 1
        dummy_model.optimize()
        loss = 0
        dummy_model.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            index = self.c.tolist().index(c[i:i+1,:].flatten().tolist())
            self.optim_model.params.ScenarioNumber = index
            loss += -1 * dummy_model.getAttr('ScenNObjVal') \
                    + 2 * torch.dot(c_pred[i:i+1,:].flatten(),torch.tensor(self.__x[index])) \
                    - self.__obj_val[index]
            dummy_model.params.ScenarioNumber += 1
        if reduction == 'sum':
            return loss
        elif reduction == 'mean':
            return loss / c_pred.size()[0]

    def SPO_loss(self, c, c_pred, reduction = 'mean'):
        '''
        :param c_pred: The outputs of the prediction model (tensor)
        :reduction: Determines whether to return total or average loss. Takes values 'mean','sum', or 'normalized'
        :return: The SPO loss value
        '''
        self.optim_model.update()
        dummy_model = self.optim_model.copy()
        dummy_model.setAttr('NumScenarios', c_pred.size()[0])
        dummy_model.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            for j in range(c_pred.size()[1]):
                dummy_model.getVars()[j].setAttr('ScenNObj',c_pred[i:i+1,:].flatten().tolist()[j])
            dummy_model.params.ScenarioNumber += 1
        dummy_model.optimize()
        loss = 0
        denom = 0
        dummy_model.params.ScenarioNumber = 0
        for i in range(c_pred.size()[0]):
            index = self.c.tolist().index(c[i:i+1,:].flatten().tolist())
            self.optim_model.params.ScenarioNumber = index
            loss += torch.dot(c[i:i+1,:].flatten(),torch.tensor(dummy_model.getAttr('ScenNX'))) \
                    - self.__obj_val[index]
            denom += self.__obj_val[index]
            dummy_model.params.ScenarioNumber += 1
        if reduction == 'sum':
            return loss
        elif reduction == 'mean':
            return loss / c_pred.size()[0]
        elif reduction == 'normalized':
            return loss / denom

    def change_model(self, X_new, c_new, train_test):
        '''
        :param X_new: The new set of input data
        :param c_new: The new outputs
        :param train_test: The index where X_new is split into train and test data
        '''
        self.X = X_new
        self.X_train = self.X[0:train_test, :]
        self.X_test = self.X[train_test:, :]
        self.c = c_new
        self.c_train = self.c[0:train_test, :]
        self.c_test = self.c[train_test:, :]
        self.optim_model.update()
        self.optim_model = self.optim_model.copy()
        self.optim_model.setAttr('NumScenarios',c_new.size()[0])
        self.optim_model.params.ScenarioNumber = 0
        for i in range(self.c.size()[0]):
            for j in range(self.c.size()[1]):
                self.optim_model.getVars()[j].setAttr('ScenNObj',self.c[i:i+1,:].flatten().tolist()[j])
            self.optim_model.params.ScenarioNumber += 1
        self.optim_model.optimize()
        self.optim_model.params.ScenarioNumber = 0
        for i in range(self.c.size()[0]):
            self.__x[i] = self.optim_model.getAttr('ScenNX')
            self.__obj_val[i] = self.optim_model.getAttr('ScenNObjVal')
            self.optim_model.params.ScenarioNumber += 1












