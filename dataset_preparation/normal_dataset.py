import pandas as pd 
import numpy as np
import argparse
import pickle
import os

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

## == Params ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tasks', type=int, default=5, help='')
parser.add_argument('--dataset', type=str, default='mini_imagenet', help='') #[mnist, fmnist, cifar10, cifar100, mini_imagenet]
parser.add_argument('--seed', type=int, default=2, help='')
parser.add_argument('--saved', type=str, default='./data/', help='')
args = parser.parse_args()

# = Add some variables to args ===
args.data_path = 'data/{}'.format(args.dataset)
args.train_file = '{}_train.csv'.format(args.dataset)
args.test_file = '{}_test.csv'.format(args.dataset)

## == Apply seed ======================
np.random.seed(args.seed)

## == Save dir ========================
if not os.path.exists(args.saved):
  os.makedirs(args.saved)


if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset == 'mnist':
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

  ## ========================================
  # == Get Cifar100 dataset =================
  if args.dataset == 'cifar100':
    cifar100_train = unpickle(os.path.join(args.data_path, 'cifar100_train'))
    cifar100_test = unpickle(os.path.join(args.data_path, 'cifar100_test'))
    X_train = np.array(cifar100_train[b'data'])
    y_train = np.array(cifar100_train[b'fine_labels'])
    X_test = np.array(cifar100_test[b'data'])
    y_test = np.array(cifar100_test[b'fine_labels'])
    
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get miniImagenet dataset =============
  if args.dataset == 'mini_imagenet':
    imagenet_train = unpickle(os.path.join(args.data_path, 'mini-imagenet-cache-train.pkl'))
    X_train = imagenet_train["image_data"]
    X_train = X_train.reshape([64, 600, 84, 84, 3])
    imagenet_val = unpickle(os.path.join(args.data_path, 'mini-imagenet-cache-val.pkl'))
    X_val = imagenet_val["image_data"]
    X_val = X_val.reshape([16, 600, 84, 84, 3])
    imagenet_test = unpickle(os.path.join(args.data_path, 'mini-imagenet-cache-test.pkl'))
    X_test = imagenet_test["image_data"]
    X_test = X_test.reshape([20, 600, 84, 84, 3])

    imagenet_data = np.concatenate((X_train, X_val, X_test), axis=0) #[100, 600, 84, 84, 3]
    imagenet_label = np.repeat(np.arange(100).reshape(-1, 1), 600, axis=1)  #[100, 600]

    X_train, X_test = np.split(imagenet_data, [540], axis=1) # 90% for train, 10% for test
    y_train, y_test = np.split(imagenet_label, [540], axis=1) # 90% for train, 10% for test
    X_train, X_test = X_train.reshape(-1, 84, 84, 3), X_test.reshape(-1, 84, 84, 3)
    y_train, y_test = y_train.flatten(), y_test.flatten()

    X_train, X_test = np.moveaxis(X_train, 3, 1), np.moveaxis(X_test, 3, 1)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print(X_train.shape)
    print(X_test.shape)


  train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
  # test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)


  pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_file),
    header=None,
    index=None
  )
  print('done')
  # pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_file),
  #   header=None,
  #   index=None
  # )
  
