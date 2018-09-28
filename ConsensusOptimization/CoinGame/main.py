import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from array2gif import write_gif

from matplotlib import pyplot as plt 
import numpy as np

import gridworld
import model
import os
import utils
import itertools

ACTIONS = [0, 1, 2, 3]

# Hyperparameters 
gamma = 0.99
learning_rate = 7e-4
# learning_rate = 0.005 

max_eps_len = 20 
eta = 3

p1_eats_own = 0
p2_eats_own = 0

p1_eats_other = 0
p2_eats_other = 0

loss = torch.tensor(0)
saving_eps = 10000

output_path = "output"

eps_collect_image = 1000
eps_collect_data = 500

p1_total_reward = 0
p2_total_reward = 0

# Define 2 Players 
p1 = model.A2C()
p2 = model.A2C()

# Make only one game 
game = gridworld.game

# Optimizer
p1_optimizer = optim.RMSprop(p1.parameters(), learning_rate, eps=0.1)
# p1_optimizer = optim.SGD(p1.parameters(), learning_rate)
p2_optimizer = optim.RMSprop(p2.parameters(), learning_rate, eps=0.1)
# p2_optimizer = optim.SGD(p2.parameters(), learning_rate)

writer = SummaryWriter(f'{output_path}/runs')

try:
    os.mkdir(f'{output_path}/images')
except FileExistsError:
    print("Directory Created")

def consensus_grad_calculate(p1, p2, p1_loss, p2_loss):
    var = [p for p in p1.parameters()] + [p for p in p2.parameters()]
    grad_player1_1 = torch.autograd.grad(p1_loss, p1.parameters(), create_graph=True) 
    grad_player2_2 = torch.autograd.grad(p2_loss, p2.parameters(), create_graph=True)  

    grad_all = grad_player1_1 + grad_player2_2

    reg = 0.5 * sum(torch.sum(g**2) for g in grad_all)
    Jgrads = torch.autograd.grad(reg, var)

    final_grad = [g + 20. * j for g, j in zip(grad_all, Jgrads)]
    return final_grad 

def LOLA_grad_calculate(p1, p2, p1_loss, p2_loss, dis_reward_both):
    total_dis_reward_1, total_dis_reward_2 = dis_reward_both

    grad_player1_1 = torch.autograd.grad(p1_loss, p1.parameters(), create_graph=True) 
    grad_player2_2 = torch.autograd.grad(p2_loss, p2.parameters(), create_graph=True)  

    update1 = [g1 + eta * total_dis_reward_2 * ((g1 * g2) + g2) for g1, g2 in zip(grad_player1_1, grad_player2_2)]
    update2 = [g2 + eta * total_dis_reward_1 * ((g1 * g2) + g1) for g1, g2 in zip(grad_player1_1, grad_player2_2)]
    
    return update1 + update2 


def calculate_loss(observation, agent, action_taken, reward_all, dis_reward_all, done, last_reward):
    loss = 0
    assert len(observation) == len(action_taken) == len(reward_all) 
    hx, cx = torch.zeros(1, 64), torch.zeros(1, 64)

    for i, (o, a, r, dr) in enumerate(zip(observation, action_taken, reward_all, dis_reward_all)):
        obs_torch = torch.from_numpy(o).unsqueeze(0).float()
        policy, pred_val, (hx, cx) = agent((obs_torch, (hx, cx)))

        if i == len(observation) - 1:
            if done:
                advantage = torch.tensor([[last_reward]])
                target_val = torch.tensor([[last_reward]])
            else:
                advantage = torch.tensor([[dr]]) - pred_val
                target_val = r + gamma * pred_val.detach()
        else:
            advantage = torch.tensor([[dr]]) - pred_val
            target_val = torch.tensor([[dr]])

        dist_policy = torch.distributions.Categorical(policy) 
        l2_loss = nn.MSELoss()

        entropy_loss = torch.sum(policy * torch.log(policy))
        policy_loss = -1 * dist_policy.log_prob(a) * advantage.detach()
        value_loss = l2_loss(pred_val, target_val)

        loss += policy_loss - 0.01 * entropy_loss + 0.5 * value_loss
        # loss += policy_loss + 0.5 * value_loss

    return loss/len(observation)

for eps in range(100000):
    # Start the game 
    obs, reward, _ = game.reset()
    p1_hx, p1_cx = torch.zeros(1, 64), torch.zeros(1, 64)
    p2_hx, p2_cx = torch.zeros(1, 64), torch.zeros(1, 64)
    
    observation = []
    p1_action_taken, p2_action_taken = [], []
    p1_reward_all, p2_reward_all = [], []

    for _ in range(max_eps_len):
        obs_torch = torch.from_numpy(obs).unsqueeze(0).float()

        p1_policy, p1_value, (p1_hx, p1_cx) = p1((obs_torch, (p1_hx, p1_cx)))
        p2_policy, p2_value, (p2_hx, p2_cx) = p2((obs_torch, (p2_hx, p2_cx)))
        
        # Sample the value
        p1_dist = torch.distributions.Categorical(p1_policy)
        p2_dist = torch.distributions.Categorical(p2_policy)

        p1_action_torch = p1_dist.sample()
        p2_action_torch = p2_dist.sample()

        p1_action = p1_action_torch.numpy()[0]
        p2_action = p2_action_torch.numpy()[0]
        
        actions = (p1_action, p2_action)
        next_obs, reward, done = game.step(actions)
        
        observation.append(obs)
        p1_action_taken.append(p1_action_torch)
        p2_action_taken.append(p2_action_torch)
        
        p1_reward, p2_reward = reward

        p1_total_reward += p1_reward
        p2_total_reward += p2_reward

        p1_reward_all.append(p1_reward)
        p2_reward_all.append(p2_reward)
       
        if done:
            if game.p1_eats_own:
                p1_eats_own += 1
            elif not game.p1_eats_own is None:
                p1_eats_other += 1
            
            if game.p2_eats_own:
                p2_eats_own += 1
            elif not game.p1_eats_own is None:
                p2_eats_other += 1
            break

        obs = next_obs

    # Training the agents, based on the performance
    p1_dis_reward_all = utils.cal_discount_reward(p1_reward_all, gamma) 
    p2_dis_reward_all = utils.cal_discount_reward(p2_reward_all, gamma)  

    p1_optimizer.zero_grad()
    p2_optimizer.zero_grad()

    p1_loss = calculate_loss(observation, p1, p1_action_taken, p1_reward_all, p1_dis_reward_all, done, float(p1_reward))
    p2_loss = calculate_loss(observation, p2, p2_action_taken, p2_reward_all, p2_dis_reward_all, done, float(p2_reward))

    final_grad = consensus_grad_calculate(p1, p2, p1_loss, p2_loss) 
    # final_grad = LOLA_grad_calculate(p1, p2, p1_loss, p2_loss, (p1_dis_reward_all[0], p2_dis_reward_all[0]))
    params = itertools.chain(p1.parameters(), p2.parameters())

    for p, g in zip(params, final_grad):
        p.grad = g
    
    # p1_loss.backward()
    nn.utils.clip_grad_norm_(p1.parameters(), 40.)
    p1_optimizer.step()
    
    # p2_loss.backward()
    nn.utils.clip_grad_norm_(p2.parameters(), 40.)
    p2_optimizer.step()

    if eps%eps_collect_image == 0:
        file_name = f'{output_path}/images/{eps}.gif'
        write_gif([gridworld.trans_image(o) for o in observation], file_name, fps=5)

    if eps%eps_collect_data == 0:
        p1_avg_reward = p1_total_reward/eps_collect_data
        p2_avg_reward = p2_total_reward/eps_collect_data

        print("-----------------------")
        print(f"At episode {eps}")
        print(f"Player 1 Loss is {p1_loss}")
        print(f"Player 2 Loss is {p2_loss}")
        
        print(f"Player 1 Avg Reward is {p1_avg_reward}")
        print(f"Player 2 Avg Reward is {p2_avg_reward}")

        print(f"Player 1 Sampled Policy {p1_policy}")
        print(f"Player 2 Sampled Policy {p2_policy}")

        print(f"Player 1 Sampled value {p1_value}")
        print(f"Player 2 Sampled value {p2_value}")
        
        print(f"Player 1 eats it own reward {p1_eats_own}")
        print(f"Player 2 eats it own reward {p2_eats_own}")
        
        print(f"Player 1 eats it other reward {p1_eats_other}")
        print(f"Player 2 eats it other reward {p2_eats_other}")

        writer.add_scalar('data/player1_loss', p1_loss, eps)
        writer.add_scalar('data/player2_loss', p2_loss, eps)
        
        writer.add_scalar('data/player1_avg_reward', p1_avg_reward, eps)
        writer.add_scalar('data/player2_avg_reward', p2_avg_reward, eps)
        
        writer.add_scalar('data/player1_own_reward', p1_eats_own, eps)
        writer.add_scalar('data/player2_own_reward', p2_eats_own, eps)
        
        writer.add_scalar('data/player1_other_reward', p1_eats_other, eps)
        writer.add_scalar('data/player2_other_reward', p2_eats_other, eps)
        
        p1_total_reward = 0 
        p2_total_reward = 0  
        
        p1_eats_own = 0
        p2_eats_own = 0

        p1_eats_other = 0
        p2_eats_other = 0

