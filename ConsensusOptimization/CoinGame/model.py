import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

class A2C(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.lstm_size = 32
        self.lstm = nn.LSTMCell(576, self.lstm_size)

        self.policy = nn.Linear(self.lstm_size, 4)
        self.value = nn.Linear(self.lstm_size, 1)
        
    def forward(self, inputs):
        x, (hx, cx) = inputs 

        batch_size = x.size(0)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(batch_size, -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        p = self.policy(x)
        return F.softmax(p, 1), self.value(x), (hx, cx)
