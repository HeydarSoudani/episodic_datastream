import os
from pandas import read_csv
from torch.utils.data import DataLoader

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from trainers.episodic_train import train as episodic_train
from trainers.batch_train import train as batch_train
from detectors.pt_detector import detector_preparation

def init_learn(model,
               learner,
               memory,
               detector,
               train_data,
               args, device):

  ### == train data ========================
  print('train_data: {}'.format(train_data.shape))  
  
  start_time = time.time()
  # == Train Model (Batch) ===========
  if args.algorithm == 'batch':
    batch_train(
      model,
      learner,
      train_data,
      args, device)

  # == Train Model (Episodic) ========
  else:
    episodic_train(
      model,
      learner,
      train_data,
      args, device)
  
  init_learning_time = time.time() - start_time
  print('init_learning time is: {%.4f}s'.format(init_learning_time))

  # == save model for plot ===========
  model.save(os.path.join(args.save, "model_after_init.pt"))
  print("= ...model after init saved")

  start_time = time.time()
  # == Save Novel detector ============
  print("Calculating detector ...")
  samples, known_labels, intra_distances\
    = detector_preparation(model,
                           learner.prototypes,
                           train_data,
                           args, device)

  detector.threshold_calculation(intra_distances,
                                 known_labels,
                                 args.std_coefficient)
  print("Detector Threshold: {}".format(detector.thresholds))  
  detector.save(args.detector_path)
  print("Detector has been saved in {}.".format(args.detector_path))
  
  ## == Save Memory selector ============
  print("Creating memory ...")
  memory.select(data=samples)
  memory.save(args.memory_path)
  print("Memory has been saved in {}.".format(args.memory_path))

  init_detector_memory_time = time.time() - start_time
  print('init detector & memory time is: {%.4f}s'.format(init_detector_memory_time))

  # ## == Test ============================
  
  # # = load data ===
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
  
  # # = 
  # print('Test with last model')
  # _, acc_dis, acc_cls = learner.evaluate(model,
  #                                       test_dataloader,
  #                                       known_labels,
  #                                       args)
  # print('Dist: {:.4f}, Cls: {}'.format(acc_dis, acc_cls))

  # print('Test with best model')
  # try: model.load(args.best_model_path)
  # except FileNotFoundError: pass
  # else: print("Load model from {}".format(args.best_model_path))
  # _, acc_dis, acc_cls = learner.evaluate(model,
  #                                       test_dataloader,
  #                                       known_labels,
  #                                       args)
  # print('Dist: {:.4f}, Cls: {}'.format(acc_dis, acc_cls))


if __name__ == '__main__':
  pass