import random
import torch
import numpy as np

def cal_discount_reward(rewards, gamma):
    """
        Calculate the discounted reward at everystep
        
        Args:
        1. rewards - list of rewards.
        Return
        1. discounted_reward - list of discounted reward
        """
    discounted_reward = []
    cumulative_sum = rewards[-1]
    for i, r in enumerate(reversed(rewards)):
        if i == 0:
            discounted_reward.append(float(r))
        else:
            cumulative_sum = (cumulative_sum + r)*gamma
            discounted_reward.append(float(cumulative_sum))
    return discounted_reward[::-1]


