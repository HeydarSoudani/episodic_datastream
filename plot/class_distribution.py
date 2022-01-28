import os
import argparse
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

def class_distribution(args):
  
  ## == load data ======================== 
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',',
    header=None).values
  labels = stream_data[:, -1].flatten()

  ## == 
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen', 'brown', 'darkcyan', 'goldenrod', 'slateblue']
  
  plt.scatter(
    np.arange(labels.shape[0]),
    labels,
    c=[colors[label] for label in labels],
    s=np.full(labels.shape, 1.4)
  )
  plt.xlabel('samples')
  plt.ylabel('class labels')

  plt.yticks(np.arange(0, 10, 1))
  plt.show()


if __name__ == '__main__':
  ## == Params ===========
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

  
