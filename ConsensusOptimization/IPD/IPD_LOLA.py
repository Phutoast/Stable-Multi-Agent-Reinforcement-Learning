# TODO - Change the update rule, and chaning the payoff.

import numpy as np
import torch
import matplotlib.pyplot as plt

import itertools
import game
import player

# inspired by https://github.com/agakshat/LOLA-pytorch/blob/master/lola-IPD-model.py
# Using the value function from LOLA paper

delta = 0.1
eta = 3 

p1 = player.Player()
p2 = player.Player()

p1_val_tracker, p2_val_tracker = [], []

for epoch in range(5000):
    # getting the first one since grad only return 1 elem tuple
    grad_player1_1, grad_player1_2 = torch.autograd.grad(game.player1ValueFunction(p1, p2), 
                                                    (p1.unnormal_policy, p2.unnormal_policy), 
                                                    create_graph=True)

    grad_player2_2, grad_player2_1 = torch.autograd.grad(game.player2ValueFunction(p1, p2), 
                                                    (p2.unnormal_policy, p1.unnormal_policy), 
                                                    create_graph=True)

    grad_player1_12 = torch.stack([torch.autograd.grad(grad_player1_1[i], p2.unnormal_policy, 
                                                    create_graph=True)[0]
                                                    for i in range(grad_player1_1.size(0))])

    grad_player2_12 = torch.stack([torch.autograd.grad(grad_player2_2[i], p1.unnormal_policy, 
                                                    create_graph=True)[0]
                                                    for i in range(grad_player2_2.size(0))])

    # According to equation 4.2
    normal_update1 = grad_player1_1  
    reg1 = torch.mm(torch.t(torch.squeeze(grad_player2_12)), grad_player1_2)
    update_grad_p1 = normal_update1 * delta + reg1 * delta * eta 

    # According to equation 4.2
    normal_update2 = grad_player2_2  
    reg2 = torch.mm(torch.t(torch.squeeze(grad_player1_12)), grad_player2_1)
    update_grad_p2 = normal_update2 * delta + reg2 * delta * eta 

    p1.unnormal_policy.data += update_grad_p1
    p2.unnormal_policy.data += update_grad_p2
    
    if epoch % 1000 == 0:
        print(epoch)
    
    if epoch % 100 == 0:
        p1_val_tracker.append(game.player1ValueFunction(p1, p2))
        p2_val_tracker.append(game.player2ValueFunction(p1, p2))

print(p1.policy)
print(p2.policy)

print(game.player1ValueFunction(p1, p2))
print(game.player2ValueFunction(p1, p2))

plt.plot(p1_val_tracker, 'C2', label='Player 1')
plt.plot(p2_val_tracker, 'C3', label='Player 2')
plt.show()
