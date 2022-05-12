import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import time

from trainers.episodic_train import train as episodic_train
from trainers.batch_train import train as batch_train
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from detectors.pt_detector import detector_preparation
from evaluation import in_stream_evaluation

def owr_test(
  model,
  prototypes,
  detector,
  current_task,
  device, args):
  
  ## == Load & prepare data ========================
  all_data = []

  task_range = args.n_task if current_task == args.n_task-1 else current_task+2
  for prev_task in range(task_range):
    task_data = pd.read_csv(
      os.path.join(args.split_test_path, "task_{}.csv".format(prev_task)),
      sep=',', header=None).values
    all_data.append(task_data)

  data = np.concatenate(all_data)
    
  if args.use_transform:
    _, test_transform = transforms_preparation()
    test_dataset = SimpleDataset(data, args, transforms=test_transform)
  else:
    test_dataset = SimpleDataset(data, args)
  dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

  ## == Test Model ==================================
  detection_results = []
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
  
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      logits, feature = model.forward(sample)

      detected_novelty, predicted_label, prob = detector(feature, prototypes)
      real_novelty = label.item() not in detector._known_labels
      detection_results.append((label.item(), predicted_label, real_novelty, detected_novelty))

      if (i+1) % 500 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s"%
          (i+1, label, predicted_label, prob, real_novelty, detected_novelty))
    
    CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
              detection_results, detector._known_labels)
    print("Evaluation: %7.2f, %7.2f, %7.2f"%(CwCA*100, M_new*100, F_new*100))


def owr_learn(
  model,
  learner,
  detector,
  memory,
  args, device):

  for task in range(args.n_tasks):  
    ### === Task data loading =====================
    task_data = pd.read_csv(
                  os.path.join(args.split_train_path, "task_{}.csv".format(task)),
                  sep=',', header=None).values 
    print('task_data: {}'.format(task_data.shape))

    ## == Split setting ===============
    if args.dataset == 'cifar100':
      if task == 5: args.ways = 20
    else: 
      if task == 1: args.ways = 4
    
    if task != 0:
      replay_mem = memory()
      train_data = np.concatenate((task_data, replay_mem))
      print('replay_mem: {}'.format(replay_mem.shape))
      print('train_data(new): {}'.format(train_data.shape))
    else:
      train_data = task_data
      print('train_data: {}'.format(train_data.shape))

    ### === Train =====================
    episodic_train(
      model,
      learner,
      train_data,
      args, device)
      
    ### === Calculating Detector ======
    print("Calculating detector ...")
    _, new_known_labels, intra_distances\
        = detector_preparation(
          model,
          learner.prototypes,
          train_data,
          args, device)
    detector.threshold_calculation(
      intra_distances,
      new_known_labels,
      args.std_coefficient)
    
    print("Detector Threshold: {}".format(detector.thresholds))
    detector.save(args.detector_path)
    print("Detector has been saved in {}.".format(args.detector_path))


    ### === Update memoty ===================
    memory.update(task_data)

    ### === After each task evaluation ======
    print('=== Testing ... ===')
    if args.which_model == 'best':
      model.load(args.best_model_path)
          
    owr_test(
      model,
      learner.prototypes,
      detector,
      task,
      device, args)


