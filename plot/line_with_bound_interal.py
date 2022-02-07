import numpy as np
import matplotlib.pyplot as plt

def get_data(dataset):

  if dataset == 'fmnist':
    ## == NTXent (InfoNCE) ===
    # data = [
    #   {
    #     'label': 'upper-bound',
    #     'mean': np.array([92.81, 92.81, 92.81, 92.81]),
    #     'std':  np.array([0.14, 0.14, 0.14, 0.14]),
    #   },
    #   {
    #     'label': 'distance-based',
    #     'mean': np.array([73.60, 80.90, 83.85, 86.12]),
    #     'std':  np.array([0.86,  0.49, 0.51, 0.26]),
    #   },
    #   {
    #     'label': 'classification-based',
    #     'mean': np.array([69.10, 79.30, 83.38, 86.42]),
    #     'std':  np.array([3.92, 1.33, 0.93, 0.38]),
    #   },
    # ]
    ## == Contrastive ========
    # data = [
    #   {
    #     'label': 'upper-bound',
    #     'mean': np.array([91.62, 91.62, 91.62, 91.62]),
    #     'std':  np.array([0.08, 0.08, 0.08, 0.08]),
    #   },
    #   {
    #     'label': 'distance-based',
    #     'mean': np.array([59.33, 68.20, 76.62, 82.60]),
    #     'std':  np.array([2.80, 3.73, 3.47, 1.67]),
    #   },
    #   {
    #     'label': 'classification-based',
    #     'mean': np.array([45.27, 58.53, 74.55, 83.47]),
    #     'std':  np.array([13.85, 11.00, 7.39, 3.01]),
    #   },
    # ]

    ## == TripletMargin ======
    data = [
      {
        'label': 'upper-bound',
        'mean': np.array([93.24, 93.24, 93.24, 93.24]),
        'std':  np.array([0.10, 0.10, 0.10, 0.10]),
      },
      {
        'label': 'distance-based',
        'mean': np.array([74.38, 80.60, 83.67, 86.33]),
        'std':  np.array([1.70, 2.24, 0.44, 0.71]),
      },
      {
        'label': 'classification-based',
        'mean': np.array([68.12, 77.97, 80.15, 84.95]),
        'std':  np.array([3.58, 2.60, 1.89, 2.25]),
      },
    ]
  
  if dataset == 'cifar10':
    ## == NTXent (InfoNCE) ===
    data = [
      {
        'label': 'upper-bound',
        'mean': np.array([77.78, 77.78, 77.78, 77.78]),
        'std':  np.array([0.73, 0.73, 0.73, 0.73]),
      },
      {
        'label': 'distance-based',
        'mean': np.array([38.52, 45.00, 48.83, 56.35]),
        'std':  np.array([0.67, 2.33, 1.15, 2.23]),
      },
      {
        'label': 'classification-based',
        'mean': np.array([22.35, 23.97, 26.52, 36.65]),
        'std':  np.array([1.81, 1.96, 0.74, 5.01]),
      },
    ]

    ## == Contrastive ========
    # data = [
    #   {
    #     'label': 'upper-bound',
    #     'mean': np.array([78.07, 78.07, 78.07, 78.07]),
    #     'std':  np.array([0.36, 0.36, 0.36, 0.36]),
    #   },
    #   {
    #     'label': 'distance-based',
    #     'mean': np.array([36.07, 45.52, 51.90, 59.15]),
    #     'std':  np.array([2.67, 2.47, 3.63, 1.27]),
    #   },
    #   {
    #     'label': 'classification-based',
    #     'mean': np.array([26.88, 34.50, 41.85, 54.85]),
    #     'std':  np.array([2.85, 3.94, 1.05, 2.04]),
    #   },
    # ]
    ## == TripletMargin ======
    # data = [
    #   {
    #     'label': 'upper-bound',
    #     'mean': np.array([80.62, 80.62, 80.62, 80.62]),
    #     'std':  np.array([0.30, 0.30, 0.30, 0.30]),
    #   },
    #   {
    #     'label': 'distance-based',
    #     'mean': np.array([33.50, 41.12, 46.50, 58.67]),
    #     'std':  np.array([2.35, 2.07, 3.10, 2.11]),
    #   },
    #   {
    #     'label': 'classification-based',
    #     'mean': np.array([21.32, 23.43, 28.30, 46.62]),
    #     'std':  np.array([2.61, 3.65, 8.96, 13.76]),
    #   },
    # ]

  if dataset == 'cifar100':
    ## == NTXent (InfoNCE) ===
    # data = [
    #   {
    #     'label': 'upper-bound',
    #     'mean': np.array([54.37, 54.37, 54.37, 54.37]),
    #     'std':  np.array([0.38, 0.38, 0.38, 0.38]),
    #   },
    #   {
    #     'label': 'distance-based',
    #     'mean': np.array([19.42, 25.38, 28.17, 30.30]),
    #     'std':  np.array([1.38, 0.28, 0.46, 0.73]),
    #   },
    #   {
    #     'label': 'classification-based',
    #     'mean': np.array([24.45, 30.68, 34.23, 37.55]),
    #     'std':  np.array([1.34,  0.19, 0.36, 0.54]),
    #   },
    # ]

    ## == Contrastive ========
    # data = [
    #   {
    #     'label': 'upper-bound',
    #     'mean': np.array([56.15, 56.15, 56.15, 56.15]),
    #     'std':  np.array([0.43, 0.43, 0.43, 0.43]),
    #   },
    #   {
    #     'label': 'distance-based',
    #     'mean': np.array([20.80, 26.42, 29.85, 31.75]),
    #     'std':  np.array([0.89, 0.51, 0.72, 0.36]),
    #   },
    #   {
    #     'label': 'classification-based',
    #     'mean': np.array([26.02, 32.12, 35.32, 38.20]),
    #     'std':  np.array([0.45, 0.61, 0.15, 0.73]),
    #   },
    # ]

    ## == TripletMargin ======
    data = [
      {
        'label': 'upper-bound',
        'mean': np.array([57.01, 57.01, 57.01, 57.01]),
        'std':  np.array([0.17, 0.17, 0.17, 0.17]),
      },
      {
        'label': 'distance-based',
        'mean': np.array([20.72, 25.85, 27.95, 30.75]),
        'std':  np.array([0.46, 0.46, 0.30, 0.23]),
      },
      {
        'label': 'classification-based',
        'mean': np.array([24.52, 30.85, 34.62, 37.15]),
        'std':  np.array([0.18, 0.65, 0.48, 0.42]),
      },
    ]

  return data

def main():
  dataset = 'cifar100' #['fmnist', 'cifar10', 'cifar100']
  metric_loss = 'TripletMargin' #['NTXent (InfoNCE)', 'Contrastive', 'TripletMargin']
  methods = ['upper-bound', 'distance-based', 'classification']
  colors = ['limegreen', 'royalblue', 'darkorange']
  x = np.arange(4)
  data = get_data(dataset)

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
  
  plt.legend(loc='lower right')
  plt.title(metric_loss)
  plt.xlabel('Memory size')
  plt.ylabel('Accuracy')
 
  # plt.xticks(np.arange(4), ['0.2K', '0.5K', '1K', '2K'])
  # plt.yticks(np.arange(30, 96, step=10)) # For FMNIST
  # plt.yticks(np.arange(10, 86, step=10)) # For CIFAR10

  # For CIFAR100
  plt.xticks(np.arange(4), ['2K', '3K', '4K', '5K'])
  plt.yticks(np.arange(10, 61, step=10)) 


  plt.show()

if __name__ == '__main__':
  main()