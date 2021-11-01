import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from math import inf
from sklearn.metrics import confusion_matrix

from datasets.dataset import DatasetFM


class ReptileDetector(object):
  def __init__(self):
    self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    self.thresholds = {}

  def __call__(self, feature):
    similarity = self.cos_sim(feature, self.weights)

    prob = -inf
    detected_novelty = True
    predicted_label = -1

    # print('_known_labels: {}'.format(self._known_labels))
    for l in self._known_labels:
      if similarity[l] >= self.thresholds[l].item() and similarity[l]>prob:
        detected_novelty = False
        predicted_label = l
        prob = similarity[l]
    
    return detected_novelty, predicted_label, prob

  def threshold_calc(self, features: list, weights):
    self.weights = weights  #[5, 768]
    self._known_labels = set([*range(0, self.weights.shape[0])])

    samples = torch.cat([item[0] for item in features])
    labels = np.array([item[1].item() for item in features])
    seen_labels = set(labels)

    # print('samples: {}'.format(samples.shape))
    # print('labels: {}'.format(len(labels)))
    # print('seen_labels: {}'.format(seen_labels))
    
    for l in seen_labels:
      sapm_per_label = samples[np.where(labels == l)[0]].squeeze()

      if sapm_per_label.dim() == 1:
        sapm_per_label = sapm_per_label.reshape(1, -1)

      # print('label: {}, sapm_per_label: {}'.format(l, sapm_per_label.shape))
      self.thresholds[l] = self.cos_sim(sapm_per_label, self.weights[l].reshape(1, -1)).mean(0)

  @property
  def known_labels(self):
    return self._known_labels

  def set_known_labels(self, label_set):
    self._known_labels = set(label_set)

  def evaluate(self, results):
    self.results = np.array(results, dtype=[
      ('true_label', np.int32),
      ('predicted_label', np.int32),
      # ('probability', np.float32),
      # ('distance', np.float32),
      ('real_novelty', np.bool),
      ('detected_novelty', np.bool)
    ])

    real_novelties = self.results[self.results['real_novelty']]
    detected_novelties = self.results[self.results['detected_novelty']]
    detected_real_novelties = self.results[self.results['detected_novelty'] & self.results['real_novelty']]

    true_positive = len(detected_real_novelties)
    false_positive = len(detected_novelties) - len(detected_real_novelties)
    false_negative = len(real_novelties) - len(detected_real_novelties)
    true_negative = len(self.results) - true_positive - false_positive - false_negative

    cm = confusion_matrix(self.results['true_label'], self.results['predicted_label'], sorted(list(np.unique(self.results['true_label']))))
    results = self.results[np.isin(self.results['true_label'], list(self._known_labels))]
    # acc = accuracy_score(results['true_label'], results['predicted_label'])
    # acc_all = accuracy_score(self.results['true_label'], self.results['predicted_label'])

    return true_positive, false_positive, false_negative, true_negative, cm

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)


def replite_detector(model, train_data, args, device):
  print('===================================== Detector =====================================')
  model.to(device)
  
  train_dataset = DatasetFM(train_data)
  dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

  novelty_detector = ReptileDetector()
  features = []
  with torch.no_grad():
    model.eval()
    for i, data in enumerate(dataloader):
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      _, feature = model.forward(sample)

      features.append((feature.detach(), label))
 
    
    weights = model.weight.detach()

  novelty_detector.threshold_calc(features, weights)

  print("detector threshold: ", novelty_detector.thresholds)
  novelty_detector.save(os.path.join(args.save, "detector.pt"))
  print("detector has been saved.")

  return novelty_detector
