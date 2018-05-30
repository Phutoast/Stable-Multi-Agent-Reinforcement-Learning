# Start with normal Matrix game Matching pennies
import numpy as np
import math
import matplotlib.pyplot as plt

# Define the learning rates
alpha = 0.8
lr_w, lr_l = 0.1, 0.05  

# Define the game 
def matchingPennies(action1, action2):
    """
    Return the reward from player1 and player2 in the game 
    matching pennies.

    Args:
        1. action1 (0 or 1) - the action of player1
        2. action2 (0 or 1) - the action of player2 

    Returns:
        1. reward1 (-1 or 1) - the reward for player1
        2. reward2 (-1 or 1) - the reward for player2

    Raises:
        1. ValueError - when the input is not 0 or not 1
    """

    # Check for the action 
    if action1 != 0 and action1 != 1:
        raise ValueError("Action for player 1 should be 0 or 1")
    
    if action2 != 0 and action2 != 1:
        raise ValueError("Action for player 2 should be 0 or 1")

    if action1 == action2:
        return 1, -1
    else:
        return -1, 1

class Player(object):
    def __init__(self):
        # Only 2 actions and one state
        self.Q = [0, 0]

        # Policy - just half.
        self.policy = [0.5, 0.5]
        self.avg_policy = [0, 0]

        # Count at state - only one 
        self.count_state = 0

    def get_expected_val(self, policy):
        return sum(q*p for q, p in zip(self.Q, policy))
    
    def sample_action(self):
        # Get the only action
        return np.random.choice(2, 1, p=self.policy)[0]

    def update_avgPolicy(self):
        for i in range(len(self.policy)):
            self.avg_policy[i] = (1/self.count_state)*(self.policy[i] - self.avg_policy[i])

    def update_policy(self, lr):
        for i in range(len(self.Q)):
            # Not sure to just update one value
            if self.Q[i] == max(self.Q):
                self.policy[i] += lr
            else:
                self.policy[i] += -lr/(len(self.Q)-1)

        # Try softmax first
        self.policy = np.exp(self.policy)/np.sum(np.exp(self.policy), axis=0) 

p1 = Player()
p2 = Player()

p1_prob_tracker = []

# Training Loop 
for i in range(500000):

    # Select action 
    act_p1 = p1.sample_action()
    act_p2 = p2.sample_action()

    p1_reward, p2_reward = matchingPennies(act_p1, act_p2)

    # Update the Q function - Simple, since no next state
    p1.Q[act_p1] = (1-alpha) * p1.Q[act_p1] + alpha*(p1_reward)    
    p2.Q[act_p2] = (1-alpha) * p2.Q[act_p2] + alpha*(p2_reward)

    # Update the average policy
    p1.count_state += 1
    p1.update_avgPolicy()
    
    p2.count_state += 1
    p2.update_avgPolicy()

    if p1.get_expected_val(p1.policy) > p1.get_expected_val(p1.avg_policy):
        p1.update_policy(lr_w)
    else:
        p1.update_policy(lr_l)

    if p2.get_expected_val(p2.policy) > p2.get_expected_val(p2.avg_policy):
        p2.update_policy(lr_w)
    else:
        p2.update_policy(lr_l)

    print("At {}".format(i))

    if i % 10000 == 0:
        p1_prob_tracker.append(p1.policy[0])

print("Policy p1 - ", p1.policy)
print("Policy p2 - ", p2.policy)

plt.plot(p1_prob_tracker)
plt.ylabel("Probability")
plt.xlabel("Time Step (10000)")
plt.show()



