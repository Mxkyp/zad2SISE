import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features=13, h1=26, h2=26, h3=26, out_features=3):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
