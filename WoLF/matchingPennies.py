# Start with normal Matrix game Matching pennies
# TODO: Generalize the code
# TODO: Better code -- Meshgrid
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

import itertools

# Define the learning rates
alpha_no_decay = 0.5

# Learning rate matter!!!
lr_w_no_decay, lr_l_no_decay = 0.001, 0.002 

def alpha_decay(t):
    return 1/(100 + t/10000)

def lr_w_decay(t):
    return 1/(20000 + t)

# Define the game here
# r_11, r_12, r_21, r_22 = 1, -1, -1, 1 
# c_11, c_12, c_21, c_22 = -1, 1, 1, -1 

r_11, r_12, r_21, r_22 = -2, 0, -3, -1 
c_11, c_12, c_21, c_22 = -2, -3, 0, -1 

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
    
    if action1 == 0 and action2 == 0:
        return r_11, c_11
    elif action1 == 0 and action2 == 1:
        return r_12, c_12 
    elif action1 == 1 and action2 == 0:
        return r_21, c_21
    elif action1 == 1 and action2 == 1:
        return r_22, c_22 

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

def test_train(init_policy_p1, init_policy_p2, save_step=1000, epoch=200000):
  p1 = Player()
  p2 = Player()
  
  p1.policy = copy.deepcopy(init_policy_p1)
  p2.policy = copy.deepcopy(init_policy_p2)
  
  del init_policy_p1
  del init_policy_p2

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
      alpha = alpha_decay(i)
      p1.Q[act_p1] = (1-alpha) * p1.Q[act_p1] + alpha*(p1_reward)    
      p2.Q[act_p2] = (1-alpha) * p2.Q[act_p2] + alpha*(p2_reward)

      # Update the average policy
      p1.count_state += 1
      p1.update_avgPolicy()

      p2.count_state += 1
      p2.update_avgPolicy()

      lr_w = lr_w_decay(i)
      lr_l = lr_w * 2

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
  
  return p1_prob_tracker, p2_prob_tracker, (p1.policy, p2.policy)

policy_test_list = [[1, 0], [0, 1], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
y_policy = [0.5, 0.5]

def plot_learning_curve(policy_test_list, y_policy,is_evol=False ,nrows=3, ncols=2):
    assert len(policy_test_list) == nrows * ncols
    final_policy_all = []
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    if is_evol:
        fig.suptitle(f"Evolution of player: where player 2 is {str(y_policy)}")
    else:
        fig.suptitle("Against Player2 - " + str(y_policy))

    for i, ax in enumerate(ax.flat):
        policy_now = policy_test_list[i]
        p1_track, p2_track, final_policy = test_train(policy_test_list[i], y_policy)
        final_policy_all.append(final_policy)

        if is_evol:
            ax.set_title("Player 1 - {}".format(policy_test_list[i]))
            ax.scatter(p1_track, p2_track, s=[i/10 for i in range(len(p1_track))])
        else:
            ax.set_title("Player 1 - {}".format(policy_test_list[i]))

            ax.plot(p1_track, 'C2', label='Player 1')
            ax.plot(p2_track, 'C3', label='Player 2')        

            ax.set_ylabel("Probability")
            ax.set_xlabel("Time Step (1000)")
            
            ax.legend()

    for init_p, final_p in zip(policy_test_list, final_policy_all):
        print(init_p, end='')
        print(" |", final_p)

    fig.tight_layout(pad=0.5)
    plt.show()

def get_gradient(p1_policy, p2_policy):
    u = r_11 - r_12 - r_21 + r_22
    u_p = c_11 - c_12 - c_21 + c_22

    grad_p1 = p2_policy * u + (r_12 - r_22) 
    grad_p2 = p1_policy * u_p + (c_12 - c_22)

    return grad_p1, grad_p2

def plot_gradient():
    # Could be better with meshgrid 
    X_cord = [i * 0.05 for i in range(0, 20)]
    Y_cord = [i * 0.05 for i in range(0, 20)]

    all_cord = list(itertools.product(X_cord, Y_cord))
    X, Y = zip(*all_cord)

    x_y_comp = [get_gradient(*xy) for xy in all_cord]
    U, V = zip(*x_y_comp)

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10, label="Gradient Field")

    plt.show()


# plot_learning_curve(policy_test_list, y_policy, nrows=3, ncols=2, is_evol=True)
plot_gradient()

