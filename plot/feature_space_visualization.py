import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from hausdorff import hausdorff_distance
from matplotlib import pyplot as plt
from pandas import read_csv

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from samplers.pt_sampler import PtSampler

def tsne_plot(features, labels, file_name='tsne'):
  # == Plot t-SNE =============================
  tsne = TSNE()
  X_embedded = tsne.fit_transform(features)

  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 6)
  sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=labels,
    legend='full',
    palette=palette
  )

  plt.savefig('{}.png'.format(file_name))
  plt.show()


def hausdorff_calculate(features, labels):
  print(labels)
  print(np.where(labels == 100)[0])
  features_novel = features[np.where(labels == 100)[0]]
  features_known = features[np.where(labels != 100)[0]]
  
  dist = hausdorff_distance(features_novel, features_known, distance="cosine")
  print('Hausdorff distance is {}'.format(dist))


def visualization(model, args, device):
  
  # == Load stream data ==============================
  # test_data = read_csv(
  #   os.path.join(args.data_path, args.dataset, args.test_file),
  #   sep=',').values
  test_data = read_csv(
    './data/{}_stream_novel.csv'.format(args.dataset),
    sep=',').values 

  if args.use_transform:
    _, test_transform = transforms_preparation()
    test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  else:
    test_dataset = SimpleDataset(test_data, args)
  
  print(test_dataset.label_set)
  print(len(test_dataset))
  sampler = PtSampler(
    test_dataset,
    n_way=6,
    n_shot=100,
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
  
  # init plot 
  model.load(os.path.join(args.save, 'model_after_init.pt'))
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

  tsne_plot(features, support_labels, file_name='tsne_init')
  hausdorff_calculate(features, support_labels)
  
  # last plot
  model.load(os.path.join(args.save, 'model_last.pt'))
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

  tsne_plot(features, support_labels, file_name='tsne_last')
  hausdorff_calculate(features, support_labels)






