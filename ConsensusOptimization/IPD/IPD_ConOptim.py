import numpy as np
import torch
import player
import game

import matplotlib.pyplot as plt

# Best - 4.5, 0.5 

delta = 4.5 
reg_param = 0.5

good_val = []

def train(d, r):
    """
    Train the algorithm with given hyperparameters 

    Params:
        1. d (float) - delta or learning rate 
        2. r (float) - regularize parameter for Jacobian
    """
    p1 = player.Player()
    p2 = player.Player()
    for epoch in range(5000):
        # getting the first one since grad only return 1 elem tuple
        grad_player1_1 = torch.autograd.grad(game.player1ValueFunction(p1, p2), p1.unnormal_policy, create_graph=True)[0]
        grad_player2_2 = torch.autograd.grad(game.player2ValueFunction(p1, p2), p2.unnormal_policy, create_graph=True)[0]
        
        # Regularizor.
        reg = 0.5 * (torch.sum(grad_player1_1**2) + torch.sum(grad_player2_2**2))
        
        Jgrad_1, Jgrad_2 = torch.autograd.grad(reg, (p1.unnormal_policy, p2.unnormal_policy))
        
        delta = d
        reg_param = r
        
        p1.unnormal_policy.data += delta * (grad_player1_1 - reg_param * Jgrad_1)
        p2.unnormal_policy.data += delta * (grad_player2_2 - reg_param * Jgrad_2)
        
        if epoch % 1000 == 0:
            print(epoch)
        
    print("Now Value")
    print(p1.policy)
    print(p2.policy)
    print(game.player1ValueFunction(p1, p2))
    print(game.player2ValueFunction(p1, p2))
    
    if game.player1ValueFunction(p1, p2) >= -30 or game.player2ValueFunction(p1, p2) >= -30:
        print("Possible delta-", d, "Reg-", r)
        return (d, r)

# Doing Grid Search
def gridSearch():
    candidate_params = []
    for r in [0.5*i for i in range(1, 21)]:
        for d in [0.5*i for i in range(1, 21)]:
            print("Now Running - Reg", r, "Delta ", d)
            poss_param = train(d, r)
            if not poss_param is None:
                candidate_params.append(poss_param)
    return candidate_params

 train(delta, reg_param)
