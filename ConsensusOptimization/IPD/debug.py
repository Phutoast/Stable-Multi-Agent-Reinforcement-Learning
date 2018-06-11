import numpy as np
import torch
import matplotlib.pyplot as plt

import itertools
import game
import player

p1 = player.Player()
p2 = player.Player()
p1.unnormal_policy = torch.t(torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]], requires_grad=True))

print(p1.policy)
print(p2.policy)
print(game.trans_matrix(p1, p2))
print("===============================")
print(game.player1ValueFunction(p1, p2))
print(game.player2ValueFunction(p1, p2))

