import torch
import pandas as pd 
import numpy as np
import argparse
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
    'fmnist',
    'cifar10'
  ],
  default='mnist',
  help='')
parser.add_argument('--seed', type=int, default=2, help='')
args = parser.parse_args()

# = Add some variables to args ===
args.data_path = 'data/{}'.format('mnist' if args.dataset=='pmnist' else args.dataset)
args.train_path = 'train'
args.test_path = 'test'
args.saved = './data/split_{}'.format(args.dataset)


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
  if args.dataset in ['mnist', 'pmnist']:
    train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Fashion-MNIST dataset ============
  if args.dataset == 'fmnist':
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
      # print(X_train.shape)
      # print(X_train.shape[-1])
      # print(X_train.size(-1))
      perm = torch.arange(X_train.shape(-1)) if t == 0 else torch.randperm(X_train.shape(-1))
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