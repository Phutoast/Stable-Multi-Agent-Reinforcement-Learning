import numpy as np
import torch
import player
import game

delta = 0.1

p1 = player.Player()
p2 = player.Player()

for epoch in range(1000):
    # getting the first one since grad only return 1 elem tuple
    grad_player1_1 = torch.autograd.grad(game.player1ValueFunction(p1, p2), p1.unnormal_policy, create_graph=True)[0]
    grad_player2_2 = torch.autograd.grad(game.player2ValueFunction(p1, p2), p2.unnormal_policy, create_graph=True)[0]
 
    p1.unnormal_policy.data += delta * grad_player1_1 
    p2.unnormal_policy.data += delta * grad_player2_2 
    
    if epoch % 100 == 0:
        print(epoch)

print(p1.policy)
print(p2.policy)
