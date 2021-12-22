import torch
import torchvision

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import argparse
import os


def imshow(imgs):
  img = torchvision.utils.make_grid(imgs)
  img = img / 2 + 0.5     # unnormalize
  npimg = img.detach().cpu().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


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


train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_test, y_test = test_data[:, 1:], test_data[:, 0]


img_view = (1, 28, 28)
X_test = train_data.view((X_test.shape[0], *img_view))

angles = [0, 10, 20, 30, 40]
for t in range(args.n_tasks):
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Rotate(angles[t]),
    transforms.ToTensor()
  ])

  X_test_aug = transform(X_test[:10])
  imshow(torch.cat([X_test[:10], X_test_aug]))
