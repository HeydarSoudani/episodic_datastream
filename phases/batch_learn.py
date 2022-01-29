import os
from pandas import read_csv
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from torch.utils.data import DataLoader

from trainers.batch_train import train

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

  train(model, train_data, test_dataloader, args, device)


  ### == Test ======================
  # if args.use_transform:
  #   _, test_transform = transforms_preparation()
  #   test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  # else:
  #   test_dataset = SimpleDataset(test_data, args)
  # test_dataloader = DataLoader(dataset=test_dataset,
  #                               batch_size=args.batch_size,
  #                               shuffle=False)

  # print('Test with last model')
  # _ = test(model, test_dataloader, args, device)

  # print('Test with best model')
  # try: model.load(args.best_model_path)
  # except FileNotFoundError: pass
  # else: print("Load model from {}".format(args.best_model_path))
  # _ = test(model, test_dataloader, args, device)




