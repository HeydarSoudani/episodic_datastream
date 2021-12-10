import os
from pandas import read_csv

from trainers.batch_train import train, test


def batch_learn(model, args, device):
  print('================================ Batch Learning =========================')
  train_data = read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',',
    header=None).values
  test_data = read_csv(
    os.path.join(args.data_path, args.test_file),
    sep=',',
    header=None).values
  
  # train(model, train_data, args, device)
  
  if args.which_model == 'best':
    try: model.load(args.best_model_path)
    except FileNotFoundError: pass
    else:
      print("Load model from {}".format(args.best_model_path))
  elif args.which_model == 'last':
    try: model.load(args.last_model_path)
    except FileNotFoundError: pass
    else:
      print("Load model from {}".format(args.last_model_path))
  
  _ = test(model, test_data, args, device)



