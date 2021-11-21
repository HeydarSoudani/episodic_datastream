import torch
import torch.nn as nn
import numpy as np
from math import inf

from datasets.dataset import SimpleDataset
from torch.utils.data import DataLoader
from utils.preparation import transforms_preparation
from utils.functions import compute_prototypes, euclidean_dist

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)

class PtDetector(object):
  def __init__(self):
    pass

  def __call__(self, feature, prototypes):
    
    pts_dict = { label: prototypes[label] for label in self._known_labels }

    detected_novelty = False
    pts = torch.cat(list(pts_dict.values()))
    labels = torch.tensor(list(pts_dict.keys()))
    dists = torch.cdist(feature.reshape(1, -1), pts).flatten()
    probs = torch.nn.functional.softmax(-dists)

    idx = torch.argmin(dists)
    min_dist = torch.min(dists)
    predicted_label = labels[idx].item()
    prob = probs[idx]

    if min_dist > self.thresholds[predicted_label]:
      detected_novelty = True
      predicted_label = -1
      prob = 0.0
      
    return detected_novelty, predicted_label, prob
  
  def set_known_labels(self, label_set):
    self._known_labels = set(label_set)
  
  def threshold_calculation(self, distances, known_labels, std_coefficient=1.0):
    self.distances = np.array(distances, dtype=[('label', np.int32), ('distance', np.float32)])
    self._known_labels = set(known_labels)
    self.std_coefficient = std_coefficient

    self.average_distances = {l: np.average(self.distances[self.distances['label'] == l]['distance'])
                            for l in self._known_labels}
    self.std_distances = {l: self.distances[self.distances['label'] == l]['distance'].std()
                        for l in self._known_labels}
    self.thresholds = {l: self.average_distances[l] + (self.std_coefficient * self.std_distances[l])
                      for l in self._known_labels}
    self.results = None

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)


def detector_preparation(model, prototypes, data, args, device):
  if args.which_model == 'best':
    model.load(args.best_model_path)

  if args.use_transform:
    _, test_transform = transforms_preparation()
    dataset = SimpleDataset(data, args, transforms=test_transform)
  else:
    dataset = SimpleDataset(data, args)
  
  dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
  known_labels = dataset.label_set


  samples = []
  intra_distances = []
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      _, feature = model.forward(sample)
      samples.append((torch.squeeze(sample, 0).detach(), label.item())) #[1, 28, 28]))
      
      # features.append((feature.detach(), label.item()))
      prototype = prototypes[label.item()]
      distance = torch.cdist(feature.detach().reshape(1, -1), prototype.reshape(1, -1))
      intra_distances.append((label, distance))

  return samples, known_labels, intra_distances



    
      
  # prototypes = compute_prototypes(features) #{label: pt, ...}
  
  # for (feature, label) in features:
  #   prototype = prototypes[label]
  #   distance = torch.cdist(feature.reshape(1, -1), prototype.reshape(1, -1))
  #   intra_distances.append((label, distance))

