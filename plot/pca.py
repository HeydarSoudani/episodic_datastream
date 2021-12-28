import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pandas import read_csv

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from samplers.pt_sampler import PtSampler


def pca(model, args, device):
  # == Load stream data ==============================
  test_data = read_csv(
    os.path.join(args.data_path, args.dataset, args.test_file),
    sep=',').values
 
  if args.use_transform:
    _, test_transform = transforms_preparation()
    test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  else:
    test_dataset = SimpleDataset(test_data, args)
  
  sampler = PtSampler(
    test_dataset,
    n_way=10,
    n_shot=500,
    n_query=0,
    n_tasks=1
  )
  test_dataloader = DataLoader(
    test_dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )

  ### ======================================
  ### == Feature space visualization =======
  ### ======================================
  print('=== Feature-Space visualization (PCA) ===')
  with torch.no_grad():
    batch = next(iter(test_dataloader))
    support_images, support_labels, _, _ = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)

    outputs, features = model.forward(support_images)
    features = features.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()
  
  pca = PCA(n_components=2)
  X_embedded = pca.fit_transform(features)

  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 10)
  sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=support_labels,
    legend='full',
    palette=palette
  )

  plt.savefig('pca.png')
  plt.show()