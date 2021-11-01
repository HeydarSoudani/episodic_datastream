import torch
from torch.utils.data import DataLoader
from pandas import read_csv

from datasets.dataset import DatasetFM
from detectors.reptile_detector import ReptileDetector
from detectors.pt_detector import PtDetector, pt_detector

from utils.preparation import transforms_preparation

def zeroshot_test(model, args, device, known_labels=None):
  print('================================ Zero-Shot Test ================================')
  # == Data ==================================
  stream_data = read_csv(args.test_path, sep=',', header=None).values
  if args.use_transform:
    _, test_transform = transforms_preparation()
    stream_dataset = DatasetFM(stream_data, transforms=test_transform)
  else:
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
  novelty_detector = PtDetector()
  if args.detector_path != '':
    novelty_detector.load(args.detector_path)
  if known_labels != None:
    novelty_detector.set_base_labels(known_labels)

  detection_results = []
  ## == Test Model ==============================
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
  
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      out, feature = model.forward(sample)

      detected_novelty, predicted_label, prob = novelty_detector(feature)
      real_novelty = label.item() not in novelty_detector.base_labels
      detection_results.append((label.item(), predicted_label, real_novelty, detected_novelty))

      if (i+1) % 500 == 0:
        print("[stream %5d]: %d, %d, %7.4f, %5s, %5s"%
          (i+1, label, predicted_label, prob, real_novelty, detected_novelty))

    tp, fp, fn, tn, cm, acc, acc_all = novelty_detector.evaluate(detection_results)
    precision = tp / (tp + fp + 1)
    recall = tp / (tp + fn + 1)
    M_new = fn / (tp + fn + 1)
    F_new = fp / (fp + tn + 1)

    print("true positive: %d"% tp)
    print("false positive: %d"% fp)
    print("false negative: %d"% fn)
    print("true negative: %d"% tn)
    print("precision: %7.4f"% precision)
    print("recall: %7.4f"% recall)
    print("M_new: %7.4f"% M_new)
    print("F_new: %7.4f"% F_new)
    print("Accuracy: %7.4f"% acc)
    print("Accuracy All: %7.4f"% acc_all)
    print("confusion matrix: \n%s"% cm)



