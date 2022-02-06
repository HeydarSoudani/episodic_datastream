import numpy as np
import matplotlib.pyplot as plt

def get_data(dataset):
  pass

def main():
  dataset = 'fmnist' #['fmnist', 'cifar10', 'cifar100']
  methods = ['upper-bound', 'distance-based', 'classification']
  colors = ['royalblue', 'hotpink', 'blueviolet']

  data = [
    {
      'label': 'upper-bound',
      'mean': np.array([]),
      'std':  np.array([]),
    },
    {
      'label': 'distance-based',
      'mean': np.array([]),
      'std':  np.array([]),
    },
    {
      'label': 'classification-based',
      'mean': np.array([]),
      'std':  np.array([]),
    },
  ]

  x = np.arange(4)

  for idx, method in enumerate(data):
    label = method['label']
    mean = method['mean']
    std = method['std']

    plt.plot(x, mean, '-o', color=colors[idx], label=label)
    plt.fill_between(
      x, mean-std, mean+std,
      edgecolor=colors[idx],
      facecolor=colors[idx],
      alpha=0.2)
  
  plt.legend(loc='lower left')
  plt.title(dataset)
  plt.xlabel('Memory size')
  plt.ylabel('Accuracy')
  plt.xticks(np.array(['0.2K', '0.5K', '1K', '2K']))
  plt.yticks(np.arange(40, 96, step=5))

  plt.show()
  