# Start with normal Matrix game Matching pennies
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

# Define the learning rate
alpha = 0.5
gamma = 0.99

# Learning rate matter!!!
lr_w, lr_l = 0.001, 0.002 

# Define the game first
def iter_prisoner(action1, action2):
    """
    Return the reward and current state(the actions) 
    in the game iterated prisoner
    Args:
        1. action1 (0 or 1) - the action of player1
            0 means betrays
            1 means coop
        2. action2 (0 or 1) - the action of player2
            0 means betrays
            1 means coop
    Returns:
        1. reward1 - the reward for player1
        2. reward2 - the reward for player2
        3. the last action(input of this method) - state
    Raises:
        1. ValueError - when the input is not 0 or not 1
    """

    if action1 != 0 and action1 != 1:
        raise ValueError("Action for player 1 should be 0 or 1")
    
    if action2 != 0 and action2 != 1:
        raise ValueError("Action for player 2 should be 0 or 1")
    
    final_reward = ()
    if action1 == 0 and action2 == 0:
        final_reward = (-2, -2)
    elif action1 == 0 and action2 == 1:
        final_reward = (0, -3) 
    elif action1 == 1 and action2 == 0:
        final_reward = (-3, 0) 
    elif action1 == 1 and action2 == 1:
        final_reward = (-1, -1)

    return final_reward, (action1, action2)

class Player(object):
    def __init__(self):
        # (-1, -1) means at the start

        self.Q = {
            (-1, -1): [0, 0],
            (0, 0): [0, 0],
            (0, 1): [0, 0],
            (1, 0): [0, 0],
            (1, 1): [0, 0],
        }

        # Policy - just half.
        self.policy = {
            (-1, -1): [0.5, 0.5],
            (0, 0): [0.5, 0.5],
            (0, 1): [0.5, 0.5],
            (1, 0): [0.5, 0.5],
            (1, 1): [0.5, 0.5],
        }
        self.avg_policy = {
            (-1, -1): [0, 0],
            (0, 0): [0, 0],
            (0, 1): [0, 0],
            (1, 0): [0, 0],
            (1, 1): [0, 0],
        }

        # Count at state 
        self.count_state = {
            (-1, -1): 0,
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0,
        }

    def get_expected_val(self, policy, state):
        return sum(q*p for q, p in zip(self.Q[state], policy[state]))

    def sample_action(self, state):
        # Get the only action
        return np.random.choice(2, 1, p=self.policy[state])[0]

    def update_avgPolicy(self, state):
        self.count_state[state] += 1

        new_avg = []
        for i, a in enumerate(self.avg_policy[state]):
            new_avg.append(1/self.count_state[state] * (self.policy[state][i] - a)) 

        self.avg_policy[state] = new_avg

    def new_lr(self, state, action, lr):
        return min(self.policy[state][action], lr) # there are only 2 actions

    def update_policy(self, state, lr):
        for i in range(len(self.Q[state])):
            if self.Q[state][i] == max(self.Q[state]):
                self.policy[state][i] +=  sum(self.new_lr(state, a, lr) for a in range(len(self.Q[state])) if a != i)
            else:
                self.policy[state][i] -= self.new_lr(state, i, lr) # Action correspond to index.

        # Try to regularized it 
        self.policy[state] = [p/sum(self.policy[state]) for p in self.policy[state]]

def test_train(save_step=1000, epoch=100000):
    p1 = Player()
    p2 = Player()
    
    # p1.policy = copy.deepcopy(init_policy_p1)
    # p2.policy = copy.deepcopy(init_policy_p2)
    # del init_policy_p1
    # del init_policy_p2

    # Doing the tracker later.
    for epoch in range(10000):
        # Running the game, fix the range to be 10 games
        for i in range(10):
            if i == 0:
                state_now = (-1, -1)

            act_p1 = p1.sample_action(state_now)
            act_p2 = p2.sample_action(state_now)

            (p1_reward, p2_reward), new_state = iter_prisoner(act_p1, act_p2)
            
            # Update Q function
            p1.Q[state_now][act_p1] = (1-alpha) * p1.Q[state_now][act_p1] + alpha*(p1_reward + gamma*p1.Q[new_state][act_p1])    
            p2.Q[state_now][act_p2] = (1-alpha) * p2.Q[state_now][act_p2] + alpha*(p2_reward + gamma*p2.Q[new_state][act_p2])

            # Update Average Policy
            p1.update_avgPolicy(state_now)
            p2.update_avgPolicy(state_now)

            if p1.get_expected_val(p1.policy, state_now) > p1.get_expected_val(p1.avg_policy, state_now):
                p1.update_policy(state_now, lr_w)
            else:
                p1.update_policy(state_now, lr_l)
            
            if p2.get_expected_val(p2.policy, state_now) > p2.get_expected_val(p2.avg_policy, state_now):
                p2.update_policy(state_now, lr_w)
            else:
                p2.update_policy(state_now, lr_l)

            state_now = new_state

    return p1.policy, p2.policy 

p1, p2 = test_train()
print(p1)
