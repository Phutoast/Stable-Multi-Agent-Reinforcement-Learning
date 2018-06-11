import torch

class Player(object):
    def __init__(self):
        self.unnormal_policy = torch.t(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True))
        self._policy = torch.sigmoid(self.unnormal_policy)
    
    @property
    def policy(self):
        return torch.sigmoid(self.unnormal_policy)
