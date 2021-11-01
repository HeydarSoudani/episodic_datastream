import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import random
import numpy as np
from pandas import read_csv
from collections import Counter

from trainers.train import train
from datasets.dataset import DatasetFM
from detectors.reptile_detector import ReptileDetector, replite_detector
from detectors.pt_detector import PtDetector, pt_detector
from utils.data_selector import DataSelector


def stream_learn(model, args, device):
  
  args.epochs = args.retrain_epochs
  args.meta_iteration = args.retrain_meta_iteration
  # class_weights = [1., 1., 1., 1., 1.]
  
  ## == Data ===================
  train_data = read_csv(args.train_path, sep=',', header=None).values
  train_dataset = DatasetFM(train_data)
  base_labels = train_dataset.label_set

  stream_data = read_csv(args.test_path, sep=',', header=None).values
  stream_dataset = DatasetFM(stream_data)
  dataloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)

  ## == Load model ============================
  if args.which_model == 'best':
    try:
      model.load(args.best_model_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.best_model_path))
  elif args.which_model == 'last':
    try:
      model.load(args.last_model_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.last_model_path))
  model.to(device)

  ## == Load Detector ==========================
  # novelty_detector = ReptileDetector()
  novelty_detector = PtDetector(base_labels)
  if args.detector_path != '':
    novelty_detector.load(args.detector_path)
    print("Load detector from file {}".format(args.detector_path))

  ## == Selector ==============================
  retrain_data_selector = DataSelector(
                            init_data=train_data,
                            model=model,
                            device=device)

  ## == Stream ================================
  buffer = [] 
  
  for i, data in enumerate(dataloader):
    model.eval()

    sample, label = data
    sample, label = sample.to(device), label.to(device)
    
    with torch.no_grad():
      _, feature = model.forward(sample)

    real_novelty = label.item() not in novelty_detector.base_labels
    detected_novelty, predicted_label, prob = novelty_detector(feature)
    
    if detected_novelty:
      sample = torch.squeeze(sample, 0)
      buffer.append((sample, label))

    print("[stream %5d]: %d, %d, %7.4f, %5s, %5s"%
      (i+1, label, predicted_label, prob, real_novelty, detected_novelty))
    
    if len(buffer) == args.buffer_size:

      new_train_data = retrain_data_selector.renew(buffer)
      # time.sleep(5)
      
      new_label_set = set([int(item[-1]) for item in new_train_data])
      print(new_label_set)
    
      # 3) Retrain Model =================
      train(model, new_train_data, args, device)

      # 3.1) Update model params. with best model
      model.load(args.best_model_path)

      # 4) Update Novel detector =========
      # novelty_detector = replite_detector(model, new_train_data, args, device)
      novelty_detector = pt_detector(
                          model,
                          new_train_data,
                          base_labels,
                          args,
                          device)
      time.sleep(3)
      buffer.clear()










      # ####
      # # 1) Create new train data (1000 from init_train + 1000 Buffer)
      # ####
      # buffer_np = torch.cat(buffer).detach().cpu().numpy()
      # train_data_sel = train_data[np.random.choice(train_data.shape[0], 1000, replace=False), :]
      # new_train_data = np.concatenate((train_data_sel, buffer_np), axis=0)
      # random.shuffle(new_train_data)

      # # 2) Convert Wa -> Wb ==============
      # buffer_label_set = set([int(item[-1]) for item in buffer_np])
      # print('buffer labels: {}'.format(buffer_label_set))

      # new_train_set = set([int(item[-1]) for item in new_train_data])
      # print('new_train labels: {}'.format(new_train_set))

      # c_list = Counter([int(item[-1]) for item in new_train_data])
      # print('class: {}'.format(c_list))
      
      # https://discuss.pytorch.org/t/how-to-dynamically-change-the-size-of-nn-linear/29634/2
      # with torch.no_grad():
      # if args.seen_labels in buffer_label_set:
      # #     model.weight = nn.Parameter(
      # #       torch.cat((model.weight, torch.randn(1, args.hidden_dims).to(device)), 0))
      # #     class_weights.append(10.0)
      #   print('yes')
      #   args.seen_labels += 1
      #   args.ways += 1
          
  
  # = retrain with last buffer
  new_train_data = retrain_data_selector.renew(buffer)
  new_label_set = set([int(item[-1]) for item in new_train_data])
  print(new_label_set)

  # 3) Retrain Model =================
  train(model, new_train_data, args, device)

  # 3.1) Update model params. with best model
  model.load(args.best_model_path)

  # 4) Update Novel detector =========
  # novelty_detector = replite_detector(model, new_train_data, args, device)
  novelty_detector = pt_detector(model, new_train_data, base_labels, args, device)
