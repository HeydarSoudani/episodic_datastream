from torch.utils.data import DataLoader
import os
import numpy as np
from pandas import read_csv

from datasets.dataset import SimpleDataset
from trainers.batch_train import train, test
from utils.preparation import transforms_preparation

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
  
  n, _ = train_data.shape
  np.random.shuffle(train_data)
  train_val_data = np.split(train_data, [int(n*0.9), n])
  train_data = train_val_data[0]
  val_data = train_val_data[1]

  train_transform, test_transform = transforms_preparation()
  if args.use_transform:
    train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
    val_dataset = SimpleDataset(val_data, args, transforms=test_transform)
    test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  else:
    train_dataset = SimpleDataset(train_data, args)
    val_dataset = SimpleDataset(val_data, args)
    test_dataset = SimpleDataset(test_data, args)

  train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
  
  # train(model, train_loader, val_loader, args, device)
  
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
  
  test(model, test_loader, device)



