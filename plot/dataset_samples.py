import torch
import torchvision
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(1, 'D:/uni/MS/_MS_thesis/codes/ml_openset')
from datasets.dataset import SimpleDataset
from samplers.pt_sampler import PtSampler


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
    'pfmnist',
    'rfmnist',
    'cifar10'
  ],
  default='pfmnist',
  help='')
parser.add_argument('--seed', type=int, default=5, help='')
args = parser.parse_args()

# = Add some variables to args ========
args.split_train_path = 'data/split_{}/train'.format(args.dataset)
args.split_test_path = 'data/split_{}/test'.format(args.dataset)


## == Apply seed ======================
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def imshow(imgs):
  # imgs *= 255.0
  grid_imgs = torchvision.utils.make_grid(torch.tensor(imgs), nrow=10)
  plt.imshow(grid_imgs.permute(1, 2, 0))
  plt.show()


def show_samples():
  
  fig, axs = plt.subplots(5, 1)

  ## incremental version
  for task in range(args.n_tasks):
    ## = Data ===========
    task_data = pd.read_csv(
      os.path.join(args.split_train_path, "task_{}.csv".format(task)),
      sep=',', header=None).values 
    dataset = SimpleDataset(task_data[:30], args)
    sampler = PtSampler(
      dataset,
      n_way=10,
      n_shot=1,
      n_query=0,
      n_tasks=1)
    dataloader = DataLoader(
      dataset,
      batch_sampler=sampler,
      num_workers=1,
      pin_memory=True,
      collate_fn=sampler.episodic_collate_fn)

    batch = next(iter(dataloader))
    support_images, support_labels, _, _ = batch
    support_images = torch.squeeze(support_images, 1)
    
    # imshow(support_images)
    grid_imgs = torchvision.utils.make_grid(torch.tensor(support_images), nrow=10)
    
    axs[task].set_title('task {}'.format(task))
    axs[task].set_xticks([])
    axs[task].set_yticks([])
    axs[task].imshow(grid_imgs.permute(1, 2, 0))
    print('task {} done!'.format(task))
  
  plt.show()

if __name__ == '__main__':
  show_samples()
