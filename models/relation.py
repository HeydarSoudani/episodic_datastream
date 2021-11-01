import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

class RelationMLP(nn.Module):
  
  def __init__(self, feature_size):
    super(RelationMLP, self).__init__()
    self.fc1 = nn.Linear(feature_size, 64)
    self.fc2 = nn.Linear(64, 1)
  
  def forward(self, x):
    out = torch.relu(self.fc1(x))
    out = torch.sigmoid(self.fc2(out))
    return out

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    self.fc1 = self.fc1.to(*args, **kwargs)
    self.fc2 = self.fc2.to(*args, **kwargs)
    return self
  
  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)


