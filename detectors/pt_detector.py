import torch
import torch.nn as nn
import numpy as np
from math import inf

from datasets.dataset import DatasetFM
from torch.utils.data import DataLoader
from utils.preparation import transforms_preparation
from utils.functions import compute_prototypes, euclidean_dist

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)

class PtDetector(object):
  def __init__(self, base_labels):
    self.base_labels = base_labels

  def __call__(self, feature):
    
    # sel_label = -1
    # min_dist = inf
    # for label, prototype in self.prototypes.items():
    #   dist = torch.cdist(feature.reshape(1, -1), prototype.reshape(1, -1))
    #   if dist < min_dist:
    #     min_dist = dist
    #     sel_label = label
    detected_novelty = False
    pts = torch.cat(list(self.prototypes.values()))
    labels = torch.tensor(list(self.prototypes.keys()))
    dists = euclidean_dist(feature.reshape(1, -1), pts).flatten()
    probs = torch.nn.functional.softmax(-dists)

    # print("pts: {}".format(pts.shape))
    # print("labels: {}".format(labels))
    # print("dists: {}".format(dists))
    # print("probs: {}".format(probs))

    idx = torch.argmin(dists)
    min_dist = torch.min(dists)
    predicted_label = labels[idx].item()
    prob = probs[idx]

    # print("idx: {}".format(idx))
    # print("min_dist: {}".format(min_dist))
    # print("prob: {}".format(prob))
    # print("predicted_label: {}".format(predicted_label))
    # print(self.thresholds)
    # print(self.thresholds[predicted_label])

    if min_dist > self.thresholds[predicted_label]:
      detected_novelty = True
      predicted_label = -1
      prob = 0.0
      
    return detected_novelty, predicted_label, prob
  
  def set_base_labels(self, label_set):
    self.base_labels = set(label_set)
  
  def threshold_calculation(self, distances, prototypes, known_labels, std_coefficient=1.0):
    self.distances = np.array(distances, dtype=[('label', np.int32), ('distance', np.float32)])
    self.prototypes = prototypes
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


def detector_preparation(model, data, args, device):
  if args.which_model == 'best':
    model.load(args.best_model_path)

  _, test_transform = transforms_preparation()
  if args.use_transform:
    dataset = DatasetFM(data, transforms=test_transform)
  else:
    dataset = DatasetFM(data)
  dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

  features = []
  samples = []
  intra_distances = []
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      _, feature = model.forward(sample)

      samples.append((torch.squeeze(sample, 0).detach(), label.item())) #[1, 28, 28]))
      features.append((feature.detach(), label.item()))

    prototypes = compute_prototypes(features) #{label: pt, ...}
   
    for (feature, label) in features:
      prototype = prototypes[label]
      distance = torch.cdist(feature.reshape(1, -1), prototype.reshape(1, -1))
      intra_distances.append((label, distance))

  return samples, prototypes, intra_distances

