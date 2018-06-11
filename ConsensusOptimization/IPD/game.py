import torch

payoff1 = torch.t(torch.tensor([[-1, -3, 0, -2]])).float()
payoff2 = torch.t(torch.tensor([[-1, 0, -3, -2]])).float()
gamma = 0.96

def trans_matrix(p1, p2):
    return torch.cat([p1.policy*p2.policy, p1.policy*(1-p2.policy), 
                        (1-p1.policy)*(p2.policy), (1-p1.policy)*(1-p2.policy)], 1)

def player1ValueFunction(p1, p2):
    P = trans_matrix(p1, p2)
    p_0 = torch.unsqueeze(P[0, :], 0) 

    inv = torch.inverse(torch.eye(4) - gamma*(P[1:, :]))
    return torch.mm(torch.mm(p_0, inv), payoff1)

def player2ValueFunction(p1, p2):
    P = trans_matrix(p1, p2)
    p_0 = torch.unsqueeze(P[0, :], 0)
    
    inv = torch.inverse(torch.eye(4) - gamma*(P[1:, :]))
    return torch.mm(torch.mm(p_0, inv), payoff2)
