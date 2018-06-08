import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as sp
from matplotlib import pyplot as plt 

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(z_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 256)
        self.l4 = torch.nn.Linear(256, 256)
        self.l5 = torch.nn.Linear(256, 256)
        self.l6 = torch.nn.Linear(256, 2)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        out = F.relu(self.l5(out))

        return self.l6(out)

class Discriminator(nn.Module):
    def __init__(self, gen_out_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(gen_out_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 256)
        self.l4 = torch.nn.Linear(256, 256)
        self.l5 = torch.nn.Linear(256, 256)
        self.l6 = torch.nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        out = F.relu(self.l5(out))

        return F.sigmoid(self.l6(out))
