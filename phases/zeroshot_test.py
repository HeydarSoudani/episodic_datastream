import torch
from torch.utils.data import DataLoader
import os
import time
from pandas import read_csv

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from evaluation import final_step_evaluation

def zeroshot_test(
  model,
  prototypes,
  detector,
  args, device,
  base_labels):
  print('================================ Zero-Shot Test ================================')
  
  # == Load stream data ==============================
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',',
    header=None).values
  if args.use_transform:
    _, test_transform = transforms_preparation()
    stream_dataset = SimpleDataset(stream_data, args, transforms=test_transform)
  else:
    stream_dataset = SimpleDataset(stream_data, args)
  dataloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)

  # # == =============================================== 
  # if known_labels != None:
  #   detector.set_known_labels(known_labels)

  ## == Test Model ===================================
  zeroshot_start_time = time.time()

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

      if (i+1) % 1000 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s"%
          (i+1, label, predicted_label, prob, real_novelty, detected_novelty))
    
    CwCA, NcCA, AcCA, OwCA, M_new, F_new = final_step_evaluation(detection_results, base_labels, detector._known_labels)
    print("Evaluation: %7.2f, %7.2f, %7.2f, %7.2f, %7.2f, %7.2f"%(CwCA*100, NcCA*100, AcCA*100, OwCA*100, M_new*100, F_new*100))
    # print("confusion matrix: \n%s"% cm)
  
  print('Zero-shot test time is: {:.4f}s'.format(time.time() - zeroshot_start_time))
    



