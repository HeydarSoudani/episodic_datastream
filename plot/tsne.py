
import torch
from torch.utils.data import DataLoader
import os
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from pandas import read_csv

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from samplers.pt_sampler import PtSampler
from samplers.reptile_sampler import ReptileSampler


def tsne(model, args, device):
  
  # == Load stream data ==============================
  test_data = read_csv(
    os.path.join(args.data_path, args.dataset, '{}_test.csv'.format(args.dataset)),
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
    n_way=10,
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

  ### ======================================
  ### == Feature space visualization =======
  ### ======================================
  print('=== Feature-Space visualization (t-SNE) ===')
  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 10)

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

  tsne = TSNE()
  X_embedded = tsne.fit_transform(features)
  sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=support_labels,
    legend='full',
    palette=palette
  )

  plt.savefig('tsne.png')
  plt.show()
  ### ======================================
