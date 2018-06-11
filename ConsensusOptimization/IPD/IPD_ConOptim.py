import numpy as np
import torch
import player
import game

import matplotlib.pyplot as plt

# Best - 0.5, 10

delta = 0.3
reg_param = 5.

p1 = player.Player()
p2 = player.Player()

p1_val_tracker, p2_val_tracker = [], []

for epoch in range(5000):
    # getting the first one since grad only return 1 elem tuple
    grad_player1_1 = torch.autograd.grad(game.player1ValueFunction(p1, p2), p1.unnormal_policy, create_graph=True)[0]
    grad_player2_2 = torch.autograd.grad(game.player2ValueFunction(p1, p2), p2.unnormal_policy, create_graph=True)[0]

    # Regularizor.
    reg = 0.5 * (torch.sum(grad_player1_1**2) + torch.sum(grad_player2_2**2))
    
    Jgrad_1, Jgrad_2 = torch.autograd.grad(reg, (p1.unnormal_policy, p2.unnormal_policy))

    p1.unnormal_policy.data += delta * (grad_player1_1 + reg_param * Jgrad_1)  
    p2.unnormal_policy.data += delta * (grad_player2_2 + reg_param * Jgrad_2) 
    
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
