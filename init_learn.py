from trainers.episodic_train import train
from detectors.pt_detector import detector_preparation

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
  memory.select(model, data=samples)
  memory.save(args.memory_path)
  print("Memory has been saved in {}.".format(args.memory_path))


if __name__ == '__main__':
  pass