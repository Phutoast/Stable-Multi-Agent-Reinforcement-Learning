import numpy as np
import torch

import itertools
import game
import player

# inspired by https://github.com/agakshat/LOLA-pytorch/blob/master/lola-IPD-model.py
# Using the value function from LOLA paper

delta = 0.5
eta = 3 

p1 = player.Player()
p2 = player.Player()

for epoch in range(1000):
    # getting the first one since grad only return 1 elem tuple
    grad_player1_1, grad_player1_2 = torch.autograd.grad(game.player1ValueFunction(p1, p2), 
                                                    (p1.unnormal_policy, p2.unnormal_policy), create_graph=True)

    grad_player2_2, grad_player2_1 = torch.autograd.grad(game.player2ValueFunction(p1, p2), 
                                                    (p2.unnormal_policy, p1.unnormal_policy), create_graph=True)

    grad_player1_12 = torch.stack([torch.autograd.grad(grad_player1_1[i], p2.unnormal_policy, create_graph=True)[0]
                                   for i in range(grad_player1_1.size(0))])
    grad_player2_12 = torch.stack([torch.autograd.grad(grad_player2_2[i], p1.unnormal_policy, create_graph=True)[0]
                                   for i in range(grad_player2_2.size(0))])


    # According to equation 4.2
    normal_update1 = grad_player1_1  
    reg1 = torch.mm(torch.t(torch.unsqueeze(grad_player1_2, 1)), grad_player2_12)
    update_grad_p1 = normal_update1 * delta + reg1 * delta * eta 

    # According to equation 4.2
    normal_update2 = grad_player2_2  
    reg2 = torch.mm(torch.t(torch.unsqueeze(grad_player2_1, 1)), grad_player1_12)
    update_grad_p2 = normal_update2 * delta + reg2 * delta * eta 

    p1.unnormal_policy.data += torch.squeeze(update_grad_p1)
    p2.unnormal_policy.data += torch.squeeze(update_grad_p2)
    
    if epoch % 100 == 0:
        print(epoch)
        print(p1.policy)
        print(p2.policy)
