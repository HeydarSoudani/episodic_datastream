from torch.utils.data import DataLoader
import os
from pandas import read_csv

from trainers.episodic_train import train
from trainers.batch_train import test
from detectors.pt_detector import detector_preparation
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation


def init_learn(model,
               pt_learner,
               memory,
               detector,
               train_data,
               args,
               device):

  ### == Train Model =================
  train(model, pt_learner, train_data, args, device)


  test_data = read_csv(
    os.path.join(args.data_path, args.test_file),
    sep=',',
    header=None).values
  
  if args.which_model == 'best':
    try: model.load(args.best_model_path)
    except FileNotFoundError: pass
    else:
      print("Load model from {}".format(args.best_model_path))
  elif args.which_model == 'last':
    try: model.load(args.last_model_path)
    except FileNotFoundError: pass
    else:
      print("Load model from {}".format(args.last_model_path))
  
  if args.use_transform:
    _, test_transform = transforms_preparation()
    test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  else:
    test_dataset = SimpleDataset(test_data, args)
  test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)

  test(model, test_dataloader, args, device)



  # ## == Save Novel detector ===========
  # print("Calculating detector ...")
  # samples, known_labels, intra_distances\
  #   = detector_preparation(model,
  #                          pt_learner.prototypes,
  #                          train_data,
  #                          args, device)

  # detector.threshold_calculation(intra_distances,
  #                                known_labels,
  #                                args.std_coefficient)
  # print("Detector Threshold: {}".format(detector.thresholds))  
  # detector.save(args.detector_path)
  # print("Detector has been saved in {}.".format(args.detector_path))
  
  
  # ## == Save Memory selector ==========
  # print("Creating memory ...")
  # memory.select(data=samples)
  # memory.save(args.memory_path)
  # print("Memory has been saved in {}.".format(args.memory_path))


if __name__ == '__main__':
  pass