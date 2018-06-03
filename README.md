# Win-or-Learn-Fast
Implementation for the paper "Multiagent Learning Using a Variable Learning Rate"

## Result

### Matching Pennies
We test the performance of agent by making it play against itself, but with difference initialization of the initial policy.

#### Against [0, 1] for 100,000 iterations  
Some converges to an optimal policy.  
Hyperparameter - alpha = 0.5, lr_w, lr_l = 0.001, 0.002

![alt-text](img/fig1.png)

| Player 1 init Policy | Final Policy (player1/player2)|
| ------------- | ------------- |
| (1.0, 0.0) |(0.4650000000000004, 0.5349999999999996) (0.5930030090069373, 0.40699699099306275)|
| (0.0, 1.0) |(0.5099999999999996, 0.49000000000000044) (0.5390000000000005, 0.4609999999999995)|
| (0.5, 0.5) |(0.505, 0.495) (0.5860010030039995, 0.4139989969960005)|
| (0.2, 0.8) |(0.4922938633542938, 0.5077061366457062) (0.5089999999999996, 0.49100000000000044)|
| (0.8, 0.2) |(0.18331535524490944, 0.8166846447550906) (0.0, 1.0)|
| (0.3, 0.7) |(0.5064423489303821, 0.4935576510696179) (0.49000000000000044, 0.5099999999999996)|

#### Against [0.5, 0.5] for 100,000 iterations
For some experiment, the policy converges, some aren't.
Hyperparameter - alpha = 0.5, lr_w, lr_l = 0.001, 0.002

Converged Policy

![alt-text](img/fig4.png)

| Player 1 init Policy | Final Policy (player1/player2)|
| ------------- | ------------- |
| (1.0, 0.0) | (0.4109999999999995, 0.5890000000000005), (0.5420000000000005, 0.4579999999999995) |
| (0.0, 1.0) | (0.5600000000000005, 0.4399999999999995), (0.4289999999999995, 0.5710000000000005) |
| (0.5, 0.5) | (0.561, 0.43899999999999995), (0.517, 0.483) |
| (0.2, 0.8) | (0.5572031872509963, 0.44279681274900373), (0.51799203187251, 0.48200796812749) |
| (0.8, 0.2) | (0.44179681274900373, 0.5582031872509963), (0.53599203187251, 0.46400796812749) |
| (0.3, 0.7) | (0.5680000000000003, 0.4319999999999997), (0.484, 0.516) |

Non-Converged Policy

![alt-text](img/fig3.png)


| Player 1 init Policy | Final Policy (player1/player2)|
| ------------- | ------------- |
| (1.0, 0.0) | (0.5731610251757697, 0.4268389748242303), (0.0, 1.0) |
| (0.0, 1.0) | (0.5660000000000005, 0.4339999999999995), (0.49699999999999955, 0.5030000000000004) |
| (0.5, 0.5) | (0.518, 0.482), (0.489, 0.511) |
| (0.2, 0.8) | (0.5750000000000003, 0.4249999999999997), (0.484, 0.516) |
| (0.8, 0.2) | (0.46299999999999975, 0.5370000000000003), (0.519, 0.481) |
| (0.3, 0.7) | (0.5647888446215142, 0.43521115537848576), (0.53599203187251, 0.46400796812749) |


### Prisoner's Dilemma
Same setting as the matching pennies. Action at position 0 means defect, and at position 1 means cooperate.

#### Against [0, 1] for 100,000 iterations
All the test cases are the same, and all converges to Nash Equilibrium(all defect).
Hyperparameter - alpha = 0.5, lr_w, lr_l = 0.001, 0.002

![alt-text](img/fig2.png)

#### Against [0.5, 0.5] for 100,000 iterations
All the test cases are the same, and all converges to Nash Equilibrium(all defect).
Hyperparameter - alpha = 0.5, lr_w, lr_l = 0.001, 0.002

![alt-text](img/fig5.png)

### Iterated Prisoner's Dilemma
The algorithm somewhat implement tit-for-tat strategy (although the start policy should be both cooperative). 

The result policy is 
Hyperparameter - alpha = 0.8, gamma = 0.99, lr_w, lr_l = 0.001, 0.002
Run for 300,00

(prob of defect, prob of coop)

| Last State (player1/player2) | Final Policy (player1/player2) |
| ---------------------------- | ------------------------------ |
| Start | (1.0, 0.0) (1.0, 0.0) |
| Defect/Defect | (1.0, 0.0) (1.0, 0.0) |
| Defect/Coop | (0.742, 0.258) (0.958, 0.042) |
| Coop/Defect | (0.9, 0.1) (1.0, 0.0) |
| Coop/Coop | (0.646, 0.354) (0.654, 0.346) | 
