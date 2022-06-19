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
    'cifar10',
    'cifar100'
  ],
  default='cifar100',
  help='')
parser.add_argument('--seed', type=int, default=1, help='')
args = parser.parse_args()

## == Apply seed ======================
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def imshow(imgs):
  # imgs *= 255.0
  grid_imgs = torchvision.utils.make_grid(torch.tensor(imgs), nrow=10)
  plt.imshow(grid_imgs.permute(1, 2, 0))
  plt.show()


def show_samples():
  
  datasets = ['mnist', 'fmnist', 'cifar10']
  dataset_titles = ['MNIST', 'FashionMNIST', 'CIFAR10']
  fig, axs = plt.subplots(args.n_tasks, len(datasets))

  for idx, dataset in enumerate(datasets):
    args.dataset = dataset
    split_train_path = 'data/split_{}/train'.format(dataset)
    split_test_path = 'data/split_{}/test'.format(dataset)

    for task in range(args.n_tasks):
      ## = Data ====================
      task_data = pd.read_csv(
        os.path.join(split_train_path, "task_{}.csv".format(task)),
        sep=',', header=None).values 
      dataset = SimpleDataset(task_data[:300], args)
      sampler = PtSampler(
        dataset,
        n_way=2,
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
      ## ============================

      # imshow(support_images)
      grid_imgs = torchvision.utils.make_grid(torch.tensor(support_images), nrow=10)
      
      if idx == 0:
        axs[task, idx].set_ylabel('task {}'.format(task+1), fontsize=12, rotation=0, ha='right')
      if task == 0:
        axs[task, idx].set_title(dataset_titles[idx], fontsize=13)
      
      axs[task, idx].set_xticks([])
      axs[task, idx].set_yticks([])
      axs[task, idx].imshow(grid_imgs.permute(1, 2, 0))
      print('task {} done!'.format(task))
    
  plt.show()

if __name__ == '__main__':
  show_samples()
