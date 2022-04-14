from gurobipy import *
from package import SPOPlus
import torch
from torch import nn

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, c_pred, c):
        return (1 - 2*torch.mul(c_pred,c)).flatten() if (1 - 2*torch.mul(c_pred,c)) >= 0 else 0


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
c_pred = torch.rand(2).view(1,-1)
print(hinge(c_pred,c[0:1,:]))
print(SPO_obj.SPO_Plus_Loss(c[0:1,:],c_pred))

