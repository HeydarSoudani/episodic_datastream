import torch
import torch.nn as nn
import numpy as np
from math import inf

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)

class PtDetector(object):
  def __init__(self, base_labels):
    self.base_labels = base_labels

  def __call__(self, feature):
    min_dist = inf
    sel_label = -1

    # pts = {label: self.prototypes[label] for label in self._known_labels}
    # for label, prototype in pts.items():
    for label, prototype in self.prototypes.items():
      dist = torch.cdist(feature.reshape(1, -1), prototype.reshape(1, -1))
      if dist < min_dist:
        min_dist = dist
        sel_label = label

    if min_dist > self.thresholds[sel_label]:
    # if min_dist > self.thresholds.get(sel_label, 0.0):
    # if torch.min(dist) > self.thresholds[torch.argmin(dist).item()]:
      detected_novelty = True
      prob = 0.0
      predicted_label = -1
    else:
      if sel_label in self.base_labels:
        detected_novelty = False
      else:
        detected_novelty = True  
      predicted_label = sel_label

      prob = 0.0
      # Calc. prob
      # dist = torch.cdist(feature.reshape(1, -1), self.prototypes)      #[1, cls_num]
      # exp_dists = (-0.1 * dist.pow(2)).exp() #[1, cls_num]
      # numerator = exp_dists[0, predicted_label]            
      # denominator = torch.sum(exp_dists, 1)
      # prob = torch.div(numerator, denominator)
    return detected_novelty, predicted_label, prob
  
  def set_base_labels(self, label_set):
    self.base_labels = set(label_set)
  
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





















  # def make_prototypes(self, data):
  #   """
  #   data: list of tuples (feature, label)
  #   output: 
  #   """
  #   features = torch.cat([item[0] for item in data])
  #   support_labels = torch.cat([item[1] for item in features])
  #   seen_labels = torch.unique(support_labels)

  #   # Prototype i is the mean of all instances of features corresponding to labels == i
  #   self.prototypes = {
  #     l.item(): support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
  #     for l in seen_labels
  #   }

  #   # self.prototypes = torch.cat(
  #   #   [
  #   #     support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
  #   #     for l in seen_labels
  #   #   ]
  #   # )
    
  #   return self.prototypes

# def pt_detector(
#   model,
#   data,
#   base_labels,
#   args,
#   device
# ):
#   print('===================================== Detector =====================================')
#   model.to(device)

#   transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#   ])
#   dataset = DatasetFM(data)
#   # dataset = DatasetFM(data, transforms=transform)
#   dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

  
#   novelty_detector = PtDetector(base_labels)
#   features = []
#   intra_distances = []
#   with torch.no_grad():
#     model.eval()
#     for i, data in enumerate(dataloader):
#       sample, label = data
#       sample, label = sample.to(device), label.to(device)
#       _, feature = model.forward(sample)
      
#       features.append((feature.detach(), label))
     
#     prototypes = novelty_detector.make_prototypes(features) #[cls, feature_num]
#     # prototypes = prototypes.to('cpu')

#     for i, data in enumerate(dataloader):
#       sample, label = data
#       sample = sample.to(device)
#       _, feature = model.forward(sample)
#       feature, label = feature.to('cpu'), label.to('cpu')

#       prototype = prototypes[label.item()].to('cpu')
#       distance = torch.cdist(feature.reshape(1, -1), prototype.reshape(1, -1))
#       intra_distances.append((label, distance))
#     novelty_detector.threshold_calc(intra_distances, dataset.label_set, args.std_coefficient)
  
#   print("distance average: {}".format(novelty_detector.average_distances))
#   print("distance std: {}".format(novelty_detector.std_distances))
#   print("detector threshold: {}".format(novelty_detector.thresholds))
  
#   novelty_detector.save(os.path.join(args.save, "detector.pt"))
#   print("detector has been saved.")

#   return novelty_detector


  # def evaluate(self, results):
  #   self.results = np.array(results, dtype=[
  #     ('true_label', np.int32),
  #     ('predicted_label', np.int32),
  #     # ('probability', np.float32),
  #     # ('distance', np.float32),
  #     ('real_novelty', np.bool),
  #     ('detected_novelty', np.bool)
  #   ])

  #   real_novelties = self.results[self.results['real_novelty']]
  #   detected_novelties = self.results[self.results['detected_novelty']]
  #   detected_real_novelties = self.results[self.results['detected_novelty'] & self.results['real_novelty']]

  #   true_positive = len(detected_real_novelties)
  #   false_positive = len(detected_novelties) - len(detected_real_novelties)
  #   false_negative = len(real_novelties) - len(detected_real_novelties)
  #   true_negative = len(self.results) - true_positive - false_positive - false_negative

  #   cm = confusion_matrix(self.results['true_label'], self.results['predicted_label'], sorted(list(np.unique(self.results['true_label']))))
  #   results = self.results[np.isin(self.results['true_label'], list(self._known_labels))]
  #   acc = accuracy_score(results['true_label'], results['predicted_label'])
  #   acc_all = accuracy_score(self.results['true_label'], self.results['predicted_label'])

  #   return true_positive, false_positive, false_negative, true_negative, cm, acc, acc_all
