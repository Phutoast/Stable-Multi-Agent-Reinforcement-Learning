# Win-or-Learn-Fast
Implementation for the paper "Multiagent Learning Using a Variable Learning Rate"

## Result 
We test the performance of agent by making it play against itself, but with difference initinalization of the initial policy. All agent converges to near optimal strategy ([0.5, 0.5]).

### Against [1, 0] for 500 iterations  

![alt-text](img/fig1.png)

| Player 1 init Policy | Final Policy (player1/player2)|
| ------------- | ------------- |
| 1.0, 0.0 | (0.48754216, 0.51245784) (0.48110016, 0.51889984) |
| 0.0, 1.0 | (0.4912146, 0.5087854) (0.48748886, 0.51251114) |
| 0.5, 0.5 | (0.51986153, 0.48013847) (0.4900016, 0.5099984) |
| 0.2, 0.8 | (0.51948208, 0.48051792) (0.48502755, 0.51497245) |
| 0.8, 0.2 | (0.48034304, 0.51965696) (0.48213117, 0.51786883) |
| 0.3, 0.7 | (0.48235767, 0.51764233) (0.49516762, 0.50483238) |
