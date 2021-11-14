import torch

from trainers import train
from detectors.pt_detector import detector_preparation
# from detectors.reptile_detector import replite_detector
# from detectors.pt_detector import pt_detector



def init_learn(model,
               memory,
               detector,
               train_data,
               base_labels,
               args,
               device):

  ### == Train Model =================
  train(model, train_data, args, device)


  ## == Save Novel detector ===========
  print("Calculating detector ...")
  samples, prototypes, intra_distances = detector_preparation(model, train_data, args, device)
  
  detector.threshold_calculation(intra_distances, prototypes, base_labels, args.std_coefficient)
  print("Detector Threshold: {}".format(detector.thresholds))  
  detector.save(args.detector_path)
  print("Detector has been saved in {}.".format(args.detector_path))
  
  
  ## == Save Memory selector ==========
  print("Creating memory ...")
  memory.select(data=samples)
  memory.save(args.memory_path)
  print("Memory has been saved in {}.".format(args.memory_path))


if __name__ == '__main__':
  pass