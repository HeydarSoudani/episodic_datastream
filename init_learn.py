from pandas import read_csv

from datasets.dataset import DatasetFM
from trainers import train
from detectors.reptile_detector import replite_detector
from detectors.pt_detector import pt_detector

def init_learn(model, args, device):

  ## == Load model if exist =========
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

  ## == load train data from file ===
  train_data = read_csv(args.train_path, sep=',', header=None).values
  train_dataset = DatasetFM(train_data)
  base_labels = train_dataset.label_set

  ### == Train Model =================
  train(model, train_data, args, device)
  # batch_train(model, train_data, args, device)

  ## == Save Novel detector =========
  if args.which_model == 'best':
    model.load(args.best_model_path)
  # _ = replite_detector(model, train_data, args, device)
  _ = pt_detector(model, train_data, base_labels, args, device)



if __name__ == '__main__':
  pass