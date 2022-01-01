import os
from pandas import read_csv
from torch.utils.data import DataLoader

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from trainers.episodic_train import train
from detectors.pt_detector import detector_preparation

def init_learn(model,
               pt_learner,
               memory,
               detector,
               train_data,
               args,
               device):

  print('train_data: {}'.format(train_data.shape))
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
  memory.select(data=samples)
  memory.save(args.memory_path)
  print("Memory has been saved in {}.".format(args.memory_path))


  ## == Test ==========================
  # test_data = read_csv(
  #           os.path.join(args.data_path, args.test_file),
  #           sep=',', header=None).values
  # print('test_data: {}'.format(test_data.shape))
  # if args.use_transform:
  #   _, test_transform = transforms_preparation()
  #   test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  # else:
  #   test_dataset = SimpleDataset(test_data, args)
  # test_dataloader = DataLoader(dataset=test_dataset,
  #                               batch_size=args.batch_size,
  #                               shuffle=False)

  # known_labels = test_dataset.label_set
  # print('Test on: {}'.format(known_labels))
  # _, acc_dis, acc_cls = pt_learner.evaluate(model,
  #                                       test_dataloader,
  #                                       known_labels,
  #                                       args)
  # print('Dist: {}, Cls: {}'.format(acc_dis, acc_cls))



if __name__ == '__main__':
  pass