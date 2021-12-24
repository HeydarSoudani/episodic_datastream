import torch
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np
import argparse
import time
import os

## == Params ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tasks', type=int, default=5, help='')
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'pmnist',
    'rmnist',
    'fmnist',
    'rfmnist',
    'cifar10'
  ],
  default='rfmnist',
  help='')
parser.add_argument('--seed', type=int, default=5, help='')
args = parser.parse_args()

# = Add some variables to args ========

if args.dataset in ['mnist', 'pmnist', 'rmnist']:
  data_folder = 'mnist'
elif args.dataset in ['fmnist', 'pfmnist', 'rfmnist']:
  data_folder = 'fmnist'
else:
  data_folder = args.dataset

args.data_path = 'data/{}'.format(data_folder)
args.saved = './data/split_{}'.format(args.dataset)
args.train_path = 'train'
args.test_path = 'test'

## == Apply seed ======================
np.random.seed(args.seed)


## == Save dir ========================
if not os.path.exists(os.path.join(args.saved, args.train_path)):
  os.makedirs(os.path.join(args.saved, args.train_path))
if not os.path.exists(os.path.join(args.saved, args.test_path)):
  os.makedirs(os.path.join(args.saved, args.test_path))


if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset in ['mnist', 'pmnist', 'rmnist']:
    train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Fashion-MNIST dataset ============
  if args.dataset in ['fmnist', 'rfmnist']:
    train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Cifar10 dataset ==================
  if args.dataset == 'cifar10':
    train_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_train.csv'), sep=',', header=None).values
    test_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_test.csv'), sep=',', header=None).values
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
  ## ========================================
  ## ========================================
  

  if args.dataset == 'pmnist':
    for t in range(args.n_tasks):
      perm = torch.arange(X_train.shape[-1]) if t == 0 else torch.randperm(X_train.shape[-1])
      # inv_perm = torch.zeros_like(perm)
      # for i in range(perm.size(0)):
      #   inv_perm[perm[i]] = i
      train_data = np.concatenate((X_train[:, perm], y_train.reshape(-1, 1)), axis=1)
      test_data = np.concatenate((X_test[:, perm], y_test.reshape(-1, 1)), axis=1)
      pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None)
      pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None)

  elif args.dataset in ['rmnist', 'rfmnist']:
    
    angles = [0, 10, 20, 30, 40]
    for t in range(args.n_tasks):
      
      if t == 0: 
        train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
        test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
        pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
        pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
      
      else:
        tensor_view = (1, 28, 28)
        rotated_xtrain_list = []
        rotated_xtest_list = []
        
        for img in X_train:
          x_tensor = (torch.tensor(img, dtype=torch.float) / 255).view(tensor_view)
          pil_img = transforms.ToPILImage()(x_tensor)
          rotated_pil_img = transforms.functional.rotate(pil_img, angles[t])
          rotated_img = transforms.ToTensor()(rotated_pil_img)
          rotated_img = rotated_img*255.0

          rotated_xtrain_list.append(rotated_img)
        rotated_xtrain = torch.stack(rotated_xtrain_list)
        rotated_xtrain = rotated_xtrain.clone().detach().numpy()
        rotated_xtrain = rotated_xtrain.reshape(rotated_xtrain.shape[0], -1)
        train_data = np.concatenate((rotated_xtrain, y_train.reshape(-1, 1)), axis=1)
        pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
        
        for img in X_test:
          x_tensor = (torch.tensor(img, dtype=torch.float) / 255).view(tensor_view)
          pil_img = transforms.ToPILImage()(x_tensor)
          rotated_pil_img = transforms.functional.rotate(pil_img, angles[t])
          rotated_img = transforms.ToTensor()(rotated_pil_img)
          rotated_img = rotated_img*255.0

          rotated_xtest_list.append(rotated_img)
        rotated_xtest = torch.stack(rotated_xtest_list)
        rotated_xtest = rotated_xtest.clone().detach().numpy()
        rotated_xtest = rotated_xtest.reshape(rotated_xtest.shape[0], -1)
        test_data = np.concatenate((rotated_xtest, y_test.reshape(-1, 1)), axis=1)
        pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)

      print('task {} dataset done!'.format(t))

  else:
    train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

    cpt = int(10 / args.n_tasks)
    for t in range(args.n_tasks):
      c1 = t * cpt
      c2 = (t + 1) * cpt
      i_tr = np.where((y_train >= c1) & (y_train < c2))[0]
      i_te = np.where((y_test >= c1) & (y_test < c2))[0]
      
      pd.DataFrame(train_data[i_tr]).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None
      )
      pd.DataFrame(test_data[i_te]).to_csv(os.path.join(args.saved, args.test_path, 'task_{}.csv'.format(t)),
        header=None,
        index=None
      )