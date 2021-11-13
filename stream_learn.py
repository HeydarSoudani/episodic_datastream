import torch
from torch.utils.data import DataLoader
import time
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
  buffer = [] 
  detection_results = []
  total_results = []
  last_idx = 0
  for i, data in enumerate(dataloader):
    
    model.eval()
    with torch.no_grad():
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      _, feature = model.forward(sample)

      real_novelty = label.item() not in detector.base_labels
      detected_novelty, predicted_label, prob = detector(feature)
      
      detection_results.append((label.item(), predicted_label, real_novelty, detected_novelty))

      if detected_novelty:
        sample = torch.squeeze(sample, 0) #[1, 28, 28]
        buffer.append((sample, label))
      
      if (i+1) % 500 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s, %d"%
          (i+1, label, predicted_label, prob, real_novelty, detected_novelty, len(buffer)))
    

    if len(buffer) == args.buffer_size:
      
      new_train_data = memory.select(buffer, return_data=True)
      train(model, new_train_data, args, device)
      
      ## == Save Novel detector ===========
      samples, prototypes, intra_distances = detector_preparation(model, new_train_data, args, device)
      new_labels = list(prototypes.keys())

      print('new_labels: {}'.format(new_labels))
      for key, pt in prototypes.items():
        print('label: {} -> pt:{}'.format(key, pt.shape))

      print("Calculating detector ...")
      detector.threshold_calculation(intra_distances,
                                     prototypes,
                                     new_labels,
                                     args.std_coefficient)
      print("Detector Threshold: {}".format(detector.thresholds))  
      detector.save(args.detector_path)
      print("Detector has been saved in {}.".format(args.detector_path))
      

      ## evaluation
      M_new, F_new, CwCA, OwCA, cm = evaluate(detection_results, new_labels)
      print("M_new: %7.4f"% M_new)
      print("F_new: %7.4f"% F_new)
      print("CwCA: %7.4f"% CwCA)
      print("OwCA: %7.4f"% OwCA)
      print("confusion matrix: \n%s"% cm)

      buffer.clear()
      detection_results.clear()
      total_results.append((i-last_idx, M_new, F_new, CwCA, OwCA))
      last_idx = i
      time.sleep(3)
  
  print(total_results)