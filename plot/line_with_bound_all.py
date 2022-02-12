import numpy as np
import matplotlib.pyplot as plt

def get_data(dataset):

  if dataset == 'fmnist':
    data = [
      # upper bounds
      {
        'label': 'Metric upper-bound',
        'mean': np.array([93.24, 93.24, 93.24, 93.24]),
        'std':  np.array([0.10, 0.10, 0.10, 0.10]),
      },
      {
        'label': 'Episodic upper-bound',
        'mean': np.array([92.51, 92.51, 92.51, 92.51]),
        'std':  np.array([0.23, 0.23, 0.23, 0.23]),
      },
      {
        'label': 'CrossEntropy upper-bound',
        'mean': np.array([91.68, 91.68, 91.68, 91.68]),
        'std':  np.array([0.13, 0.13, 0.13, 0.13]),
      },

      # Episodic approaches
      {
        'label': 'Episodic+PT',
        'mean': np.array([78.97, 84.87, 87.03, 88.63]),
        'std':  np.array([1.27, 0.46, 0.45, 0.39]),
      },
      {
        'label': 'Episodic+Reptile',
        'mean': np.array([79.23, 83.66, 86.77, 88.87]),
        'std':  np.array([1.01, 0.52, 0.12, 0.12]),
      },
      {
        'label': 'Episodic+CE',
        'mean': np.array([80.00, 84.77, 87.27, 89.13]),
        'std':  np.array([1.23, 0.29, 0.21, 0.12]),
      },

      # Metric approaches
      {
        'label': 'TripletMargin',
        'mean': np.array([74.38, 80.60, 83.67, 86.33]),
        'std':  np.array([1.70, 2.24, 0.44, 0.71]),
      },
      {
        'label': 'NTXent (InfoNCE)',
        'mean': np.array([73.60, 80.90, 83.85, 86.12]),
        'std':  np.array([0.86, 0.49, 0.51, 0.26]),
      },
      {
        'label': 'Contrastive',
        'mean': np.array([59.33, 68.20, 76.62, 82.60]),
        'std':  np.array([2.80, 3.73, 3.47, 1.67]),
      },

      # Baselines
      {
        'label': 'CoPE',
        'mean': np.array([75.92, 79.14, 81.84, 83.78]),
        'std':  np.array([0.65, 4.99, 2.79, 2.20]),
      },
      {
        'label': 'Reservoir',
        'mean': np.array([72.61, 79.72, 83.63, 83.25]),
        'std':  np.array([1.16, 3.01, 1.22, 0.54]),
      },
      {
        'label': 'MIR',
        'mean': np.array([66.0, 74.8, 72.2, 75.5]),
        'std':  np.array([4.0, 1.7, 5.0, 2.4]),
      },

      # CrossEntropy
      {
        'label': 'CrossEntropy',
        'mean': np.array([55.57, 70.30, 78.40, 83.13]),
        'std':  np.array([2.37, 2.04, 0.41, 1.44]),
      },
    ]
  
  if dataset == 'cifar10':
    data = [
      # upper bounds
      {
        'label': 'Metric upper-bound',
        'mean': np.array([80.62, 80.62, 80.62, 80.62]),
        'std':  np.array([0.30, 0.30, 0.30, 0.30]),
      },
      {
        'label': 'Episodic upper-bound',
        'mean': np.array([75.63, 75.63, 75.63, 75.63]),
        'std':  np.array([0.005, 0.005, 0.005, 0.005]),
      },
      {
        'label': 'CrossEntropy upper-bound',
        'mean': np.array([70.38, 70.38, 70.38, 70.38]),
        'std':  np.array([0.57, 0.57, 0.57, 0.57]),
      },

      # Episodic approaches
      {
        'label': 'Episodic+PT',
        'mean': np.array([41.67, 50.70, 57.13, 61.43]),
        'std':  np.array([0.25, 0.41, 2.02, 0.98]),
      },
      {
        'label': 'Episodic+Reptile',
        'mean': np.array([42.80, 50.47, 58.30, 65.03]),
        'std':  np.array([1.85, 0.99, 0.54, 0.74]),
      },
      {
        'label': 'Episodic+CE',
        'mean': np.array([41.47, 51.07, 56.90, 61.07]),
        'std':  np.array([0.95, 0.54, 0.75, 2.15]),
      },

      # Metric approaches
      {
        'label': 'TripletMargin',
        'mean': np.array([33.50, 41.12, 46.50, 58.67]),
        'std':  np.array([2.35, 2.07, 3.10, 2.11]),
      },
      {
        'label': 'NTXent (InfoNCE)',
        'mean': np.array([38.52, 45.00, 48.83, 56.35]),
        'std':  np.array([0.67, 2.33, 1.15, 2.23]),
      },
      {
        'label': 'Contrastive',
        'mean': np.array([36.07, 45.52, 51.90, 59.15]),
        'std':  np.array([2.67, 2.47, 3.63, 1.27]),
      },

      # Baselines
      {
        'label': 'CoPE',
        'mean': np.array([29.47, 33.93, 31.84, 28.71]),
        'std':  np.array([5.27, 1.75, 3.58, 5.80]),
      },
      {
        'label': 'Reservoir',
        'mean': np.array([37.06, 46.06, 46.75, 49.90]),
        'std':  np.array([0.49, 6.20, 15.31, 5.80]),
      },
      {
        'label': 'MIR',
        'mean': np.array([25.8, 32.3, 36.5, 45.2]),
        'std':  np.array([1.0, 1.4, 1.3, 2.1]),
      },

      # CrossEntropy
      {
        'label': 'CrossEntropy',
        'mean': np.array([19.87, 33.53, 41.43, 49.60]),
        'std':  np.array([0.76, 1.97, 2.70, 1.40]),
      },
    ]

  if dataset == 'cifar100':
    data = [
      # upper bounds
      {
        'label': 'Metric upper-bound',
        'mean': np.array([57.01, 57.01, 57.01, 57.01]),
        'std':  np.array([0.17, 0.17, 0.17, 0.17]),
      },
      {
        'label': 'Episodic upper-bound',
        'mean': np.array([49.05, 49.05, 49.05, 49.05]),
        'std':  np.array([0.58, 0.58, 0.58, 0.58]),
      },
      {
        'label': 'CrossEntropy upper-bound',
        'mean': np.array([58.33, 58.33, 58.33, 58.33]),
        'std':  np.array([0.11, 0.11, 0.11, 0.11]),
      },

      # Episodic approaches
      # {
      #   'label': 'Episodic+PT',
      #   'mean': np.array([]),
      #   'std':  np.array([]),
      # },
      # {
      #   'label': 'Episodic+Reptile',
      #   'mean': np.array([]),
      #   'std':  np.array([]),
      # },
      # {
      #   'label': 'Episodic+CE',
      #   'mean': np.array([]),
      #   'std':  np.array([]),
      # },

      # Metric approaches
      {
        'label': 'TripletMargin',
        'mean': np.array([24.52, 30.85, 34.62, 37.15]),
        'std':  np.array([0.18, 0.65, 0.48, 0.42]),
      },
      {
        'label': 'NTXent (InfoNCE)',
        'mean': np.array([24.45, 30.68, 34.23, 37.55]),
        'std':  np.array([1.34, 0.19, 0.36, 0.54]),
      },
      {
        'label': 'Contrastive',
        'mean': np.array([26.02, 32.12, 35.32, 38.20]),
        'std':  np.array([0.45, 0.61, 0.15, 0.73]),
      },

      # Baselines
      {
        'label': 'CoPE',
        'mean': np.array([27.40, 31.32, 34.74, 36.31]),
        'std':  np.array([0.08, 0.65, 0.38, 0.49]),
      },
      {
        'label': 'Reservoir',
        'mean': np.array([11.82, 14.37, 15.34, 19.38]),
        'std':  np.array([0.86, 3.39, 0.69, 0.46]),
      },
      {
        'label': 'MIR',
        'mean': np.array([13.00, 17.70, 19.00, 21.00]),
        'std':  np.array([0.90, 0.50, 0.90, 1.90]),
      },

      # CrossEntropy
      {
        'label': 'CrossEntropy',
        'mean': np.array([23.15, 28.75, 33.20, 36.05]),
        'std':  np.array([0.82, 0.98, 0.67, 0.53]),
      },
    ]

  return data

def main():
  dataset = 'cifar100' #['fmnist', 'cifar10', 'cifar100']
  methods = [
    'Metric upper-bound',
    'Episodic upper-bound',
    'CrossEntropy upper-bound',

    # 'Episodic+PT',
    # 'Episodic+Reptile',
    # 'Episodic+CE',
    
    'TripletMargin',
    'NTXent (InfoNCE)',
    'Contrastive',

    'CoPE',
    'Reservoir',
    'MIR'

    'CrossEntropy'
  ]
  colors = [
    'springgreen',
    'seagreen',
    'olivedrab',
  
    # 'steelblue',
    # 'deepskyblue',
    # 'darkturquoise',

    'darkorange',
    'goldenrod',
    'orange',

    'darkorchid',
    'slateblue',
    'mediumorchid',

    'red'
  ]
  
  x = np.arange(4)
  data = get_data(dataset)

  for idx, method in enumerate(data):
    label = method['label']
    mean = method['mean']
    std = method['std']

    plt.plot(x, mean, '-o', color=colors[idx], label=label)
    # plt.fill_between(
    #   x, mean-std, mean+std,
    #   edgecolor=colors[idx],
    #   facecolor=colors[idx],
    #   alpha=0.2)
  
  # plt.legend(loc='lower right')
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, fancybox=True, shadow=False)
  plt.title('All Methods')
  plt.xlabel('Memory size')
  plt.ylabel('Accuracy')
 
  # plt.xticks(np.arange(4), ['0.2K', '0.5K', '1K', '2K'])
  # plt.yticks(np.arange(55, 96, step=5)) # For FMNIST
  # plt.yticks(np.arange(15, 86, step=10)) # For CIFAR10

  # For CIFAR100
  plt.xticks(np.arange(4), ['2K', '3K', '4K', '5K'])
  plt.yticks(np.arange(10, 61, step=10)) 


  plt.show()

if __name__ == '__main__':
  main()
  