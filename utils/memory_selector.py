import time
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset import SimpleDataset

def euclidean_dist(x, y):
  '''
  Compute euclidean distance between two tensors
  '''
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)


class OperationalMemory():
  def __init__(self,
                per_class,
                novel_acceptance,
                device,
                args,
                selection_method='soft_rand'):
    self.per_class = per_class
    self.novel_acceptance = novel_acceptance
    self.device = device
    self.args = args
    self.selection_method = selection_method
    self.class_data = None

  def select(self, model, data, return_data=False):
    """
    Compute ...
    Args:
      data: list of (sample, label)
    Returns:
      ---
    """ 
    samples = torch.stack([item[0] for item in data])
    labels = torch.tensor([item[1] for item in data])
    seen_labels = torch.unique(labels)

    new_class_data = {
      l.item(): samples[(labels == l).nonzero(as_tuple=True)[0]]
      for l in seen_labels
    }

    if self.class_data != None:
      # should add buffer data
      keys = set(torch.tensor(list(self.class_data.keys())).tolist() + \
      torch.tensor(list(new_class_data.keys())).tolist())
      known_keys = set(torch.tensor(list(self.class_data.keys())).tolist())
      new_keys = set(torch.tensor(list(new_class_data.keys())).tolist())

      for key in keys:
        if key in known_keys:
          if key in new_keys:
            self.class_data[key] = torch.cat((self.class_data[key], new_class_data[key]), 0)
        else:
          self.class_data[key] = new_class_data[key]
    else:
      self.class_data = new_class_data  

    if self.selection_method == 'rand':
      self.rand_selection()
    elif self.selection_method == 'soft_rand':
      self.soft_rand_selection(model)
    
    # for label, features in self.class_data.items():
    #   print('{} -> {}'.format(label, features.shape))

    if return_data:
      returned_data_list = []
      for label, samples in self.class_data.items():
        n = samples.shape[0]
        if n >= self.novel_acceptance:
          samples = samples.reshape(samples.shape[0], -1)*255
          labels = torch.full((n, 1), label, device=self.device, dtype=torch.float) #[200, 1]
          data = torch.cat((samples, labels), axis=1)
          returned_data_list.append(data)
      
      returned_data = torch.cat(returned_data_list, 0)
      returned_data = returned_data.detach().cpu().numpy()
      np.random.shuffle(returned_data)
      
      return returned_data

  def rand_selection(self):
    for label, samples in self.class_data.items():
      n = samples.shape[0]
      if n >= self.per_class:
        idxs = np.random.choice(range(n), size=self.per_class, replace=False)
        self.class_data[label] = samples[idxs]
  
  def soft_rand_selection(self, model):


    for label, samples in self.class_data.items():

      features_list = []
      # === Preparing data ===============
      n = samples.shape[0]
      labels = torch.full((n, 1), label, device=self.device, dtype=torch.float) #[200, 1]
      data = torch.cat((samples, labels), axis=1)
      
      _, test_transform = transforms_preparation()
      if self.args.use_transform:
        dataset = SimpleDataset(data, args, transforms=test_transform)
      else:
        dataset = SimpleDataset(data, args)     
  
      dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
      model.eval()
      with torch.no_grad():
        for i, data in enumerate(dataloader):
          samples, _ = data
          samples, _ = samples.to(device)
          _, features = model.forward(samples)
          features_list.append(features)

        features = torch.cat(features_list)
      

      if n >= self.per_class:
        prototype = features.mean(0).reshape(1, -1)

        dist = euclidean_dist(features, prototype) #[n, 1]
        dist = np.squeeze(dist.detach().cpu().numpy())
        score = np.maximum(dist, 1.0001)
        score = np.log2(score)
        score /= np.sum(score)
        idxs = np.random.choice(range(n), size=self.per_class, p=score, replace=False)
        self.class_data[label] = samples[idxs]

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)


class IncrementalMemory():
  def __init__(self,
              device,
              args,
              selection_type='fixed_mem',   # ['fixed_mem', 'pre_class']
              total_size=1000,
              per_class=100,
              selection_method='soft_rand'):
    
    self.selection_type = selection_type
    self.total_size = total_size
    self.per_class = per_class
    self.selection_method = selection_method
    self.class_data = {}
  
  def __call__(self):
    return np.concatenate(list(self.class_data.values()), axis=0)

  def update(self, model, data):
    
    new_samples = np.array(data)
    labels = np.array(data[:, -1]).flatten()
    unique_labels = list(np.unique(labels))
    print('unique_labels: {}'.format(unique_labels))

    for l in unique_labels:
      self.class_data[l] = new_samples[np.where(labels == l)[0]]

    # == Calculate class size =====
    if not self.class_data:
      class_size = int(self.total_size / len(unique_labels))
    else:
      known_labels = list(self.class_data.keys())
      all_labels = unique_labels + known_labels
      class_size = int(self.total_size / len(all_labels))
    
    # == Selection ================
    if self.selection_method == 'rand':
      self.rand_selection()
    elif self.selection_method == 'soft_rand':
      self.soft_rand_selection(model)
    
  def rand_selection(self):
    for label, samples in self.class_data.items():
      n = samples.shape[0]
      if n >= self.per_class:
        idxs = np.random.choice(range(n), size=self.per_class, replace=False)
        self.class_data[label] = samples[idxs]
  
  def soft_rand_selection(self, model):

    for label, samples in self.class_data.items():
      features_list = []
      # === Preparing data ===============
      n = samples.shape[0]
      labels = torch.full((n, 1), label, device=self.device, dtype=torch.float) #[200, 1]
      data = torch.cat((samples, labels), axis=1)
      
      _, test_transform = transforms_preparation()
      if self.args.use_transform:
        dataset = SimpleDataset(data, args, transforms=test_transform)
      else:
        dataset = SimpleDataset(data, args)
      dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
      
      # === Calculate feature ===========
      model.eval()
      with torch.no_grad():
        for i, data in enumerate(dataloader):
          samples, _ = data
          samples, _ = samples.to(device)
          _, features = model.forward(samples)
          features_list.append(features)

        features = torch.cat(features_list)
      
      # === Select data ==================
      if n >= self.per_class:
        prototype = features.mean(0).reshape(1, -1)

        dist = euclidean_dist(features, prototype) #[n, 1]
        dist = np.squeeze(dist.detach().cpu().numpy())
        score = np.maximum(dist, 1.0001)
        score = np.log2(score)
        score /= np.sum(score)
        idxs = np.random.choice(range(n), size=self.per_class, p=score, replace=False)
        self.class_data[label] = samples[idxs]

  








 # for label in unique_labels:
    #   n = new_class_data[label].shape[0]
    #   idxs = np.random.choice(range(n), size=class_size, replace=False)
    #   self.class_data[label] = new_samples[idxs]
  
    # new_class_data = {
    #   l: new_samples[np.where(labels == l)[0]]
    #   for l in unique_labels
    # }
    # for label, samples in self.class_data.items():
    #   n = samples.shape[0]
    #   idxs = np.random.choice(range(n), size=class_size, replace=False)
    #   self.class_data[label] = samples[idxs]
    # if self.selection_type == 'fixed_mem':
    # elif self.selection_type == 'pre_class':
    #   for label in unique_labels:
    #     n = new_class_data[label].shape[0]
    #     idxs = np.random.choice(range(n), size=self.per_class, replace=False)
    #     self.class_data[label] = new_samples[idxs]
