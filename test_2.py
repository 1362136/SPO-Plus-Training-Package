from gurobipy import *
from package import SPOPlus
import torch
from torch import nn

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, c_pred, c):
        loss = 0
        for i in range(c_pred.size()[0]):
            loss += (1 - 2*torch.mul(c_pred[i:i+1,:],c[i:i+1,:]))\
                if (1 - 2*torch.mul(c_pred[i:i+1,:],c[i:i+1,:])) >= 0 else 0
        return loss


model = Model()
x = model.addVar(lb=-0.5,ub=0.5,obj=1.0, name='x')
model.addConstr(x <= 0.5)
model.addConstr(x >= -0.5)
model.setObjective(1*x, GRB.MINIMIZE)
model.NumScenarios = 10
model.params.scenarioNumber = 0
for j in range(10):
    x.setAttr('ScenNObj',1)
    model.params.scenarioNumber += 1

SPO_obj = SPOPlus.SPOPlus(torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]),model,5)
c = SPO_obj.get_output()
hinge = HingeLoss()
c_pred = torch.rand(2).view(2,-1)
print(hinge(c_pred,c[0:2,:]))
print(SPO_obj.SPO_Plus_Loss(c[0:2,:],c_pred[0:2,:],reduction='sum'))
SPO_obj.change_model(torch.tensor([[1],[2],[3],[4]]),torch.tensor([[1],[-1],[1],[-1]]),2)

