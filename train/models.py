import torch
import torch.nn as nn


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 1024)
        self.linear2 = torch.nn.Linear(1024, 128)
        self.linear3 = torch.nn.Linear(128, D_out)
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out1 = self.linear1(x)  # .clamp(min=0)
        out2 = self.relu1(out1)
        out3 = self.linear2(out2)
        out4 = self.relu2(out3)
        out5 = self.linear3(out4)
        out = self.sigmoid(out5)
        return out
