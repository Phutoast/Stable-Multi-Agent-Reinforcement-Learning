import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as sp
from matplotlib import pyplot as plt 

import itertools
import gridworld

game = gridworld.game
obs, reward, _ = game.reset()
obs, reward, done = game.step((3, 2))
obs, reward, done = game.step((3, 1))

print(done)
print(reward)
