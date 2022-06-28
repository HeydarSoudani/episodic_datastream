from turtle import left
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import seaborn as sns
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hausdorff import hausdorff_distance

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from samplers.pt_sampler import PtSampler


def set_novel_label(known_labels, args, data=[]):
  print(data.shape)
  
  if data == []:
    data = read_csv(
      os.path.join(args.data_path, args.stream_file),
      sep=',', header=None).values

  for idx, item in enumerate(data):
    label = item[-1]
    # print(known_labels)
    # print(label)
    if label not in known_labels:
      data[idx, -1] = 100

  return data


def tsne_plot(features, labels, file_name='tsne', n_color=6):
  
  colors = [
    'royalblue', 'forestgreen',
    'darkorchid', 'brown',
    'gold', 'red',
    'darkcyan', 'greenyellow',
    'peru',
    'hotpink'
  ]
  
  tsne = TSNE()
  X_embedded = tsne.fit_transform(features)
  
  fig, ax = plt.subplots()
  plt_colors = np.array([colors[9] if i==100 else colors[i] for i in labels])
  ax.tick_params(
    left=False,
    labelleft=False,
    bottom = False,
    labelbottom=False
  )
  scatter = ax.scatter(
    X_embedded[:,0],
    X_embedded[:,1],
    marker='o',
    c=plt_colors,
    label=['Novel' if i==100 else i for i in labels]
  )

  print(np.unique(plt_colors))
  legend1 = ax.legend(
    np.unique(plt_colors),
    loc="upper right",
    title="Classes",
    fontsize=9
  )
  ax.add_artist(legend1)
    
  # sns.set(rc={'figure.figsize':(11.7,8.27)})
  # palette = sns.color_palette("bright", n_color)
  # sns.scatterplot(
  #   x=X_embedded[:,0],
  #   y=X_embedded[:,1],
  #   hue=labels,
  #   legend='full',
  #   palette=palette
  # )
  # sns.tick_params(bottom=False, left=False)

  plt.savefig('{}.png'.format(file_name))
  # plt.show()
  plt.clf()


def pca_plot(features, labels, file_name='pca'):
  pca = PCA(n_components=2)
  X_embedded = pca.fit_transform(features)

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
  features_novel = features[np.where(labels == 100)[0]]
  features_known = features[np.where(labels != 100)[0]]
  
  dist = hausdorff_distance(features_novel, features_known, distance="cosine")
  print('Hausdorff distance is {}'.format(dist))


def visualization(model, data, args, device, filename, n_label=6):  
  
  if args.use_transform:
    _, test_transform = transforms_preparation()
    dataset = SimpleDataset(data, args, transforms=test_transform)
  else:
    dataset = SimpleDataset(data, args)
  
  print(dataset.label_set)
  print(len(dataset))
  sampler = PtSampler(
    dataset,
    n_way=n_label,
    n_shot=500,
    n_query=0,
    n_tasks=1
  )
  dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )
  
  ### == Plot ============================
  with torch.no_grad():
    batch = next(iter(dataloader))
    support_images, support_labels, _, _ = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)

    outputs, features = model.forward(support_images)
    features = features.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()

    # for feature in features:
    #   print(feature)
    # print(support_labels)
    # print(features.shape)
    # print(support_labels.shape)
  # features += 1e-12

  tsne_plot(features, support_labels, file_name=filename, n_color=n_label)
  # pca_plot(features, support_labels, file_name='pca_last')
  hausdorff_calculate(features, support_labels)






