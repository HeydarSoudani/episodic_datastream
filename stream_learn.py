import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from pandas import read_csv

from trainers.train import train
from detectors.pt_detector import detector_preparation
from datasets.dataset import DatasetFM
from evaluation import evaluate


def stream_learn(model,
                 memory,
                 detector,
                 args,
                 device):
  args.epochs = args.retrain_epochs
  args.meta_iteration = args.retrain_meta_iteration
  print('================================ Stream Learning ================================')
  ## == Data ==================================
  stream_data = read_csv(args.test_path, sep=',', header=None).values
  stream_dataset = DatasetFM(stream_data)
  dataloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)

  ## == Stream ================================
  unknown_buffer = [] 
  known_buffer = {i:[] for i in detector._known_labels}

  detection_results = []
  total_results = []
  last_idx = 0
  for i, data in enumerate(dataloader):
    
    model.eval()
    with torch.no_grad():
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      _, feature = model.forward(sample)
      real_novelty = label.item() not in detector._known_labels
      detected_novelty, predicted_label, prob = detector(feature)      
      detection_results.append((label.item(), predicted_label, real_novelty, detected_novelty))

      sample = torch.squeeze(sample, 0) #[1, 28, 28]
      if detected_novelty:
        unknown_buffer.append((sample, label))
      else:
        known_buffer[predicted_label].append((sample, label))

      if (i+1) % 100 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s, %d"%
          (i+1, label, predicted_label, prob, real_novelty, detected_novelty, len(unknown_buffer)))
    

    if (i+1) % args.known_retrain_interval == 0:
      
      known_buffer_list = []
      for label , data in known_buffer.items():
        n = len(data)
        idxs = np.random.choice(range(n), size=args.known_per_class, replace=False)
        known_buffer_list.extend([data[i] for i in idxs])
      
      
      ## 2) Preparing retrain data
      new_train_data = memory.select(known_buffer_list, return_data=True)
      
      ## 3) Retraining Model
      train(model, new_train_data, args, device)
      
      ## 4) Recalculating Detector
      print("Calculating detector ...")
      samples, prototypes, intra_distances = detector_preparation(model, new_train_data, args, device)
      new_labels = list(prototypes.keys())

      detector.threshold_calculation(intra_distances,
                                     prototypes,
                                     new_labels,
                                     args.std_coefficient)
      print("Detector Threshold: {}".format(detector.thresholds))  
      detector.save(args.detector_path)
      print("Detector has been saved in {}.".format(args.detector_path))
    
      known_buffer = {i:[] for i in detector._known_labels}

    if len(unknown_buffer) == args.buffer_size:
      sample_num = i-last_idx

      ## 1) evaluation
      M_new, F_new, CwCA, OwCA, cm = evaluate(detection_results, detector._known_labels)
      print("[On %5d samples]: %7.4f, %7.4f, %7.4f, %7.4f"%
        (sample_num, CwCA, OwCA, M_new, F_new))
      print("confusion matrix: \n%s"% cm)

      ## 2) Preparing retrain data
      new_train_data = memory.select(unknown_buffer, return_data=True)
      
      ## 3) Retraining Model
      train(model, new_train_data, args, device)
      
      ## 4) Recalculating Detector
      print("Calculating detector ...")
      samples, prototypes, intra_distances = detector_preparation(model, new_train_data, args, device)
      new_labels = list(prototypes.keys())

      detector.threshold_calculation(intra_distances,
                                     prototypes,
                                     new_labels,
                                     args.std_coefficient)
      print("Detector Threshold: {}".format(detector.thresholds))  
      detector.save(args.detector_path)
      print("Detector has been saved in {}.".format(args.detector_path))
    

      unknown_buffer.clear()
      detection_results.clear()
      total_results.append((i-last_idx, M_new, F_new, CwCA, OwCA))
      last_idx = i
      time.sleep(3)
  
  print(total_results)