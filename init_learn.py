import torch

from trainers import train
from detectors.pt_detector import detector_preparation
# from detectors.reptile_detector import replite_detector
# from detectors.pt_detector import pt_detector



def init_learn(model,
               pt_learner,
               memory,
               detector,
               train_data,
               args,
               device):

  ### == Train Model =================
  train(model, pt_learner, train_data, args, device)


  ## == Save Novel detector ===========
  print("Calculating detector ...")
  print(pt_learner.prototypes)
  samples, known_labels, intra_distances\
    = detector_preparation(model,
                           pt_learner.prototypes,
                           train_data,
                           args, device)

  detector.threshold_calculation(intra_distances,
                                 known_labels,
                                 args.std_coefficient)
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