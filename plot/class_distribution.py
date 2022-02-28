import os
import argparse
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

def class_distribution(args):
  
  seeds = ['1', '5', '12', '39']
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen', 'brown', 'darkcyan', 'goldenrod', 'slateblue']
  
  plt.rcParams['figure.figsize'] = [18, 4]
  fig, axs = plt.subplots(1, len(seeds))
  for i in range(len(seeds)):
    stream_data = read_csv(
      os.path.join(args.data_path, '{}_stream_s{}.csv'.format(args.dataset, seeds[i])),
      sep=',',
      header=None).values
    labels = stream_data[:, -1].flatten()

    axs[i].scatter(
      np.arange(labels.shape[0]),
      labels,
      c=[colors[label] for label in labels],
      s=np.full(labels.shape, 1.4)
    )
    axs[i].set_title('Seed {}'.format(seeds[i]))
    axs[i].set_xlabel('samples')
    axs[i].set_ylabel('class labels')
    axs[i].set_yticks(np.arange(0, 10, 1))
  
  plt.savefig('class_dist.png', dpi=800)
  plt.show()





  ## == load data ======================== 
  # stream_data = read_csv(
  #   os.path.join(args.data_path, args.stream_file),
  #   sep=',',
  #   header=None).values
  # labels = stream_data[:, -1].flatten()

  # ## == 
  
  # plt.scatter(
  #   np.arange(labels.shape[0]),
  #   labels,
  #   c=[colors[label] for label in labels],
  #   s=np.full(labels.shape, 1.4)
  # )
  # plt.xlabel('samples')
  # plt.ylabel('class labels')
  # plt.yticks(np.arange(0, 10, 1))
  # plt.savefig('class_dist.png')
  # plt.show()


if __name__ == '__main__':
  ## == Params ==========================
  parser = argparse.ArgumentParser()
  parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'fmnist',
    'cifar10',
    'cifar100'
  ],
  default='mnist',
  help='') 
  args = parser.parse_args()

  ## == Add some variables to args =======
  args.data_path = 'data/'
  args.train_file = '{}_train.csv'.format(args.dataset)
  args.test_file = '{}_test.csv'.format(args.dataset)
  args.stream_file = '{}_stream.csv'.format(args.dataset)

  class_distribution(args)

  
