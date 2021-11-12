import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, args):
    super(MLP, self).__init__()
    self.args = args
    self.device = None
  
  def forward(self, samples):
    pass

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0] # store device
    
    return self

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)


