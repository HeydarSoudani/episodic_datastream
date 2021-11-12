import torch

from trainers import train
# from detectors.reptile_detector import replite_detector
# from detectors.pt_detector import pt_detector

from datasets.dataset import DatasetFM
from torch.utils.data import DataLoader
from utils.preparation import transforms_preparation
from utils.functions import compute_prototypes

def init_learn(model,
               memory,
               detector,
               train_data,
               base_labels,
               args,
               device):

  ### == Train Model =================
  train(model, train_data, args, device)

  # = Extract features for updating selector and detector ===
  if args.which_model == 'best':
    model.load(args.best_model_path)

  _, test_transform = transforms_preparation()
  if args.use_transform:
    dataset = DatasetFM(train_data, transforms=test_transform)
  else:
    dataset = DatasetFM(train_data)
  dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

  features = []
  intra_distances = []
  with torch.no_grad():
    model.eval()
    for i, data in enumerate(dataloader):
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      _, feature = model.forward(sample)
      features.append((feature.detach(), label.item()))

    prototypes = compute_prototypes(features) #{label: pt, ...}
   
    for (feature, label) in features:
      prototype = prototypes[label]
      distance = torch.cdist(feature.reshape(1, -1), prototype.reshape(1, -1))
      intra_distances.append((label, distance))


  ## == Save Memory selector ==========
  print("Creating memory ...")
  memory.select(data=features)
  memory.save(args.memory_path)
  print("Memory has been saved in {}.".format(args.memory_path))


  ## == Save Novel detector ===========
  print("Calculating detector ...")
  detector.threshold_calculation(intra_distances, base_labels, args.std_coefficient)
  print("Detector Threshold: {}".format(detector.thresholds))  
  detector.save(args.detector_path)
  print("Detector has been saved in {}.".format(args.detector_path))





  # _ = replite_detector(model, train_data, args, device)
  # _ = pt_detector(model, train_data, base_labels, args, device)



if __name__ == '__main__':
  pass