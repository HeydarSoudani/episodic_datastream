import torch
from torch.utils.data import DataLoader
from pandas import read_csv

from datasets.dataset import SimpleDataset
from evaluation import evaluate
from utils.preparation import transforms_preparation

def zeroshot_test(model, detector, known_labels, args, device):
  print('================================ Zero-Shot Test ================================')
  
  # == Load stream data ==============================
  stream_data = read_csv(args.test_path, sep=',', header=None).values
  if args.use_transform:
    _, test_transform = transforms_preparation()
    stream_dataset = SimpleDataset(stream_data, args, transforms=test_transform)
  else:
    stream_dataset = SimpleDataset(stream_data, args)
  dataloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)


  detection_results = []
  ## == Test Model ===================================
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
  
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      out, feature = model.forward(sample)

      detected_novelty, predicted_label, prob = detector(feature)
      real_novelty = label.item() not in detector.base_labels
      detection_results.append((label.item(), predicted_label, real_novelty, detected_novelty))

      if (i+1) % 1000 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s"%
          (i+1, label, predicted_label, prob, real_novelty, detected_novelty))
    
    M_new, F_new, CwCA, OwCA, cm = evaluate(detection_results, known_labels)
    print("M_new: %7.4f"% M_new)
    print("F_new: %7.4f"% F_new)
    print("CwCA: %7.4f"% CwCA)
    print("OwCA: %7.4f"% OwCA)
    print("confusion matrix: \n%s"% cm)
    



