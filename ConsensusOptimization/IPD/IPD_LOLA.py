import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as sp
from matplotlib import pyplot as plt

import itertools

# inspired by https://github.com/agakshat/LOLA-pytorch/blob/master/lola-IPD-model.py
# Using the value function from LOLA paper

gamma = 0.96
delta = 0.1
eta = 0.3

class Player(object):
    def __init__(self):
        self.unnormal_policy = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self._policy = torch.sigmoid(self.unnormal_policy)
    
    @property
    def policy(self):
        return torch.sigmoid(self.unnormal_policy)

p1 = Player()
p2 = Player()

payoff1 = torch.t(torch.tensor([[-1, -3, 0, -2]])).float()
payoff2 = torch.t(torch.tensor([[-1, 0, -3, -2]])).float()

def trans_matrix(p1, p2):
    return torch.stack([
                        p1.policy*p2.policy,
                        p1.policy*(1-p2.policy),
                        (p1.policy-1)*(p2.policy),
                        (1-p1.policy)*(1-p2.policy),
                        ])


def player1ValueFunction(p1, p2):
    P = trans_matrix(p1, p2)
    p_0 = torch.unsqueeze(P[:, 0], 0)
    
    inv = torch.inverse(torch.eye(4) - gamma*P[:, 1:])
    return torch.mm(torch.mm(p_0, inv), payoff1)

def player2ValueFunction(p1, p2):
    P = trans_matrix(p1, p2)
    p_0 = torch.unsqueeze(P[:, 0], 0)
    
    inv = torch.inverse(torch.eye(4) - gamma*P[:, 1:])
    return torch.mm(torch.mm(p_0, inv), payoff2)

for epoch in range(1000):
    if epoch % 100 == 0:
        print(epoch)

    # getting the first one since grad only return 1 elem tuple
    grad_player1_1 = torch.autograd.grad(player1ValueFunction(p1, p2), p1.unnormal_policy, create_graph=True)[0]
    grad_player1_2 = torch.autograd.grad(player1ValueFunction(p1, p2), p2.unnormal_policy, create_graph=True)[0]

    grad_player2_2 = torch.autograd.grad(player2ValueFunction(p1, p2), p2.unnormal_policy, create_graph=True)[0]
    grad_player2_1 = torch.autograd.grad(player2ValueFunction(p1, p2), p1.unnormal_policy, create_graph=True)[0]

    grad_player1_12 = torch.stack([torch.autograd.grad(grad_player1_1[i], p2.unnormal_policy, retain_graph=True)[0]
                                   for i in range(grad_player1_1.size(0))])

    grad_player2_12 = torch.stack([torch.autograd.grad(grad_player2_2[i], p1.unnormal_policy, retain_graph=True)[0]
                                   for i in range(grad_player2_2.size(0))])


    # According to equation 4.2
    update_grad_p1 = grad_player1_1 * delta + torch.mm(torch.t(torch.unsqueeze(grad_player1_2, 1)), grad_player2_12)*delta*eta
    
    # According to equation 4.2
    update_grad_p2 = grad_player2_2 * delta + torch.mm(torch.t(torch.unsqueeze(grad_player2_1, 1)), grad_player1_12)*delta*eta
    
    p1.unnormal_policy.data += torch.squeeze(update_grad_p1)
    p2.unnormal_policy.data += torch.squeeze(update_grad_p2)


print(p1.policy)
print(p2.policy)

