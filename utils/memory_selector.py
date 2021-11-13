from datasets.dataset import DatasetFM

import time
import torch
import random
import numpy as np

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
                selection_method='rand'):
    self.per_class = per_class
    self.novel_acceptance = novel_acceptance
    self.device = device
    self.selection_method = selection_method
    self.class_data = None

  def select(self, data, return_data=False):
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

      print(keys)
      print(known_keys)

      for key in keys:
        if key in known_keys:
          if key in new_keys:
            self.class_data[key] = torch.cat((self.class_data[key], new_class_data[key]), 0)
        else:
          self.class_data[key] = new_class_data[key]
    else:
      self.class_data = new_class_data  
    

    for label, features in self.class_data.items():
      print('{} -> {}'.format(label, features.shape))


    if self.selection_method == 'rand':
      self.rand_selection()
    # elif self.selection_method == 'soft_rand':
    #   self.soft_rand_selection()
    
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
  
  def soft_rand_selection(self):
    for label, features in self.class_data.items():
      n = features.shape[0]
      if n >= self.per_class:
        prototype = features.mean(0).reshape(1, -1)

        dist = euclidean_dist(features, prototype) #[n, 1]
        dist = np.squeeze(dist.detach().cpu().numpy())
        score = np.maximum(dist, 1.0001)
        score = np.log2(score)
        score /= np.sum(score)
        idxs = np.random.choice(range(n), size=self.per_class, p=score, replace=False)
        self.class_data[label] = features[idxs]

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)







#['rand', 'soft_rand', ]
class DataSelector():
  def __init__(self,
                init_data,
                model,
                min_per_class=125,
                max_per_class=250,
                device='cpu',
                selection_method='soft_rand'):
    self.min_per_class = min_per_class
    self.max_per_class = max_per_class
    self.selection_method = selection_method
    self.model = model
    self.device = device

    init_dataset = DatasetFM(init_data)
    data = init_dataset.data
    self.labels_set = set(init_dataset.labels)

    ## == split data by classes =================
    self.data_class = {}
    for class_label in self.labels_set:
      self.data_class[class_label] = []
    for idx, (sample, label) in enumerate(data):
      self.data_class[label.item()].append(sample)

    ## == select data from pool data ============
    if self.selection_method == 'rand':
      self.rand_selection()
    elif self.selection_method == 'soft_rand':
      self.soft_rand_selection()
  
    # for class_label in labels_set:
    #   print('label: {}, {}'.format(class_label, len(data_class[class_label])))
  def rand_selection(self):
    for class_label in self.labels_set:
      data_len = len(self.data_class[class_label])
      
      if data_len >= self.max_per_class:
        self.data_class[class_label] = random.sample(self.data_class[class_label], self.max_per_class)
     
  def soft_rand_selection(self):
    for class_label in self.labels_set:
      n = len(self.data_class[class_label])

      if n >= self.max_per_class:
        
        data = torch.stack(self.data_class[class_label])
        with torch.no_grad():
          _, features = self.model(data.to(self.device))
        prototype = features.mean(0).reshape(1, -1)
        
        dist = euclidean_dist(features, prototype) #[n, 1]
        dist = np.squeeze(dist.detach().cpu().numpy())
        score = np.maximum(dist, 1.0001)
        score = np.log2(score)
        score /= np.sum(score)

        idxs = np.random.choice(range(n), size=self.max_per_class, p=score, replace=False)
        self.data_class[class_label] = [self.data_class[class_label][idx] for idx in idxs]
        
  
  def renew(self, new_data):
    """
    model: 
    new_data: list of tuples [(sample, label)]
    """
    new_data = [(item[0].to('cpu'), item[1].to('cpu')) for item in new_data]
    new_labels = [item[1].item() for item in new_data]
    new_labels_set = set(new_labels)
    print('buffer labels_set: {}'.format(new_labels_set))
    
    ## == split data by classes =================
    new_data_class = {}
    for class_label in new_labels_set:
      new_data_class[class_label] = []
    for idx, (sample, label) in enumerate(new_data):
      new_data_class[label.item()].append(sample)

    
    ## == New data added to store list ==========
    for new_label in new_labels_set:
      if new_label in self.labels_set:
        self.data_class[new_label].extend(new_data_class[new_label])
      else:
        self.labels_set.add(new_label)
        self.data_class[new_label] = new_data_class[new_label]
    
    print('== All data ====')
    for class_label in self.labels_set:
      print('label:{}, {}'.format(class_label, len(self.data_class[class_label])))

    ## == Renew lists ===========================
    if self.selection_method == 'rand':
      self.rand_selection()
    elif self.selection_method == 'soft_rand':
      self.soft_rand_selection()
    
    ## == Create output =========================
    return_data = []
    for class_label in self.labels_set:
      n = len(self.data_class[class_label])

      if n >= self.min_per_class:
        samples = self.data_class[class_label]
        samples = torch.cat([item.flatten().reshape(1, -1)*255 for item in samples], axis=0) #[200, 784]
        labels = torch.full((n, 1), class_label, dtype=torch.float) #[200, 1]
        data = torch.cat((samples, labels), axis=1)
        return_data.append(data)
    
    return_data = torch.cat(return_data, axis=0)
    return_data = return_data.detach().cpu().numpy()
    np.random.shuffle(return_data)

    # np.savetxt("foo.csv", return_data, delimiter=",")
    # print('write in file')
    time.sleep(3)

    return return_data
 





  
