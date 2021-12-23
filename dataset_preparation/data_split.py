import torch
import torchvision.transforms as transforms
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
    'rmnist',
    'fmnist',
    'cifar10'
  ],
  default='mnist',
  help='')
parser.add_argument('--seed', type=int, default=2, help='')
args = parser.parse_args()

# = Add some variables to args ========
args.data_path = 'data/{}'.format('mnist' if args.dataset in ['pmnist', 'rmnist'] else args.dataset)
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
  if args.dataset in ['mnist', 'pmnist', 'rmnist']:
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

  elif args.dataset == 'rmnist':
    img_view = (1, 28, 28)
    topil_trans = transforms.ToPILImage()
    totensor_trans = transforms.ToTensor()
  
    X_train = X_train.reshape((X_train.shape[0], *img_view))
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = X_test.reshape((X_test.shape[0], *img_view))
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    angles = [0, 20, 40, 60, 80]
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
        rotated_xtrain_list = []
        rotated_xtest_list = []
        
        for img in X_train:
          rotated_img = transforms.functional.rotate(topil_trans(img), angles[t])
          rotated_img = totensor_trans(rotated_img)
          rotated_img = (rotated_img*255)
          rotated_xtrain_list.append(rotated_img)
        rotated_xtrain = torch.stack(rotated_xtrain_list)
        rotated_xtrain = rotated_xtrain.cpu().detach().numpy()
        rotated_xtrain = rotated_xtrain.reshape(rotated_xtrain.shape[0], -1)
        train_data = np.concatenate((rotated_xtrain, y_train.reshape(-1, 1)), axis=1)
        pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_path, 'task_{}.csv'.format(t)),
          header=None,
          index=None)
        
        for img in X_test:
          rotated_img = transforms.functional.rotate(topil_trans(img), angles[t])
          rotated_img = totensor_trans(rotated_img)
          rotated_img = (rotated_img*255)
          rotated_xtest_list.append(rotated_img)
        rotated_xtest = torch.stack(rotated_xtest_list)
        rotated_xtest = rotated_xtest.cpu().detach().numpy()
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