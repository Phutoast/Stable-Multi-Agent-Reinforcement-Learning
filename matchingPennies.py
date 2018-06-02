# Start with normal Matrix game Matching pennies
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

# Define the learning rates
alpha = 0.5

# Learning rate matter!!!
lr_w, lr_l = 0.001, 0.002 

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
        self.policy = [0.8, 0.2]
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
            self.avg_policy[i] += (1/self.count_state)*(self.policy[i] - self.avg_policy[i])

    def new_policy(self, a, lr):
        return min(self.policy[a], lr/(len(self.Q)-1))

    def update_policy(self, lr):
        for i in range(len(self.Q)):
            change = self.new_policy(i, lr)
            # Not sure to just update one value
            if self.Q[i] == max(self.Q):
                self.policy[i] += sum(self.new_policy(a, lr) for a in range(len(self.Q))if a != i) 
            else:
                self.policy[i] -= change
        # Try to regularized it 
        self.policy = [p/sum(self.policy) for p in self.policy]

def test_train(init_policy_p1, init_policy_p2, save_step=1000, epoch=100000):
  p1 = Player()
  p2 = Player()
  
  p1.policy = copy.deepcopy(init_policy_p1)
  p2.policy = copy.deepcopy(init_policy_p2)

  # Getting the start policy for reference.
  p1_prob_tracker = [p1.policy[0]]
  p2_prob_tracker = [p2.policy[0]]

  # Training Loop 
  for i in range(epoch):

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
      
      if i % 10000 == 0:
          print("At {}".format(i))
      if i % save_step == 0:
          p1_prob_tracker.append(p1.policy[0])
          p2_prob_tracker.append(p2.policy[0])
          
  print("Policy p1 - ", p1.policy)
  print("Policy p2 - ", p2.policy)
  
  return p1_prob_tracker, p2_prob_tracker
        
policy_test_list = [[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]

fig, ax = plt.subplots(nrows=3, ncols=2)
fig.suptitle("Against Player2 - [0, 1]")
counter = 0

for i, row in enumerate(ax):
    for j, col in enumerate(row):
        policy_now = policy_test_list[counter]
        p1_track, p2_track = test_train(policy_test_list[counter], [0, 1])
        col.set_title("Player 1 - {}".format(policy_test_list[counter]))
        
        col.plot(p1_track, 'C2', label='Player 1')
        col.plot(p2_track, 'C3', label='Player 2')
        col.set_ylabel("Probability")
        col.set_xlabel("Time Step (1000)")
        
        col.legend()
        counter += 1
fig.tight_layout(pad=0.5)
plt.show()
