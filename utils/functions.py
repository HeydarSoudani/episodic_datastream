import torch
import torchvision
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def imshow(imgs):
  img = torchvision.utils.make_grid(imgs)
  # img = img / 2 + 0.5     # unnormalize
  npimg = img.detach().cpu().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def set_novel_label(args):
  train_data = pd.read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',', header=None).values
  train_labels = train_data[:, -1]
  seen_label = set(train_labels)
  
  stream_data = pd.read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',', header=None).values

  for idx, data in enumerate(stream_data):
    label = data[-1]
    if label not in seen_label:
      stream_data[idx, -1] = 100

  new_data_file = './data/{}_stream_novel.csv'.format(args.dataset)
  pd.DataFrame(stream_data).to_csv(
    new_data_file,
    header=None, index=None)


if __name__ == '__main__':
  pass