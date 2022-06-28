import numpy as np
import matplotlib.pyplot as plt

def get_data(dataset):
  if dataset == 'MNIST':
    data = [
      # upper bounds
      {
        'label': 'Metric upper-bound',
        'mean': np.array([97.75, 97.75, 97.75, 97.75]),
        'std':  np.array([0.03, 0.03, 0.03, 0.03]),
      },
      {
        'label': 'Episodic upper-bound',
        'mean': np.array([97.62, 97.62, 97.62, 97.62]),
        'std':  np.array([0.07, 0.07, 0.07, 0.07]),
      },
      {
        'label': 'CE upper-bound',
        'mean': np.array([97.78, 97.78, 97.78, 97.78]),
        'std':  np.array([0.05, 0.05, 0.05, 0.05]),
      },

      # Meta approaches
      {
        'label': 'ML-PT',
        'mean': np.array([84.10, 90.60, 92.95, 94.62]),
        'std':  np.array([0.78, 0.16, 0.30, 0.15]),
      },
      {
        'label': 'ML-RP',
        'mean': np.array([84.38, 90.45, 92.98, 95.70]),
        'std':  np.array([0.38, 0.23, 0.29, 0.14]),
      },
      {
        'label': 'ML-CE',
        'mean': np.array([85.25, 90.60, 92.95, 94.62]),
        'std':  np.array([]),
      },

      # Metric approaches
      {
        'label': 'MT-NX',
        'mean': np.array([70.15, 83.05, 87.30, 91.40]),
        'std':  np.array([]),
      },
      {
        'label': 'MT-CO',
        'mean': np.array([63.85, 79.10, 81.00, 90.27]),
        'std':  np.array([]),
      },
      {
        'label': 'MT-TR',
        'mean': np.array([57.40, 75.70, 84.12, 89.55]),
        'std':  np.array([]),
      },
      
      # Baselines
      {
        'label': 'CoPE',
        'mean': np.array([80.37, 86.80, 91.15, 93.00]),
        'std':  np.array([]),
      },
      {
        'label': 'Reservoir',
        'mean': np.array([79.09, 86.78, 90.24, 91.59]),
        'std':  np.array([]),
      },
      {
        'label': 'MIR',
        'mean': np.array([86.70, 92.70, 94.30, 95.30]),
        'std':  np.array([]),
      },
      {
        'label': 'GSS',
        'mean': np.array([47.44, 66.08, 86.20, 92.23]),
        'std':  np.array([]),
      },
      {
        'label': 'iCaRL',
        'mean': np.array([71.87, 72.67, 73.18, 73.24]),
        'std':  np.array([]),
      },
    ]
  
  if dataset == 'FashionMNIST':
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
        'label': 'CE upper-bound',
        'mean': np.array([91.68, 91.68, 91.68, 91.68]),
        'std':  np.array([0.13, 0.13, 0.13, 0.13]),
      },

      # Episodic approaches
      {
        'label': 'ML-PT',
        'mean': np.array([76.53, 83.45, 87.03, 88.90]),
        'std':  np.array([]),
      },
      {
        'label': 'ML-RP',
        'mean': np.array([77.98, 82.15, 86.15, 88.97]),
        'std':  np.array([]),
      },
      {
        'label': 'ML-CE',
        'mean': np.array([77.83, 84.25, 87.20, 88.95]),
        'std':  np.array([]),
      },

      # Metric approaches
      {
        'label': 'MT-NX',
        'mean': np.array([73.60, 80.90, 83.85, 86.12]),
        'std':  np.array([0.86, 0.49, 0.51, 0.26]),
      },
      {
        'label': 'MT-CO',
        'mean': np.array([59.33, 68.20, 76.62, 82.60]),
        'std':  np.array([2.80, 3.73, 3.47, 1.67]),
      },
      {
        'label': 'MT-TR',
        'mean': np.array([74.38, 80.60, 83.67, 86.33]),
        'std':  np.array([1.70, 2.24, 0.44, 0.71]),
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
      {
        'label': 'GSS',
        'mean': np.array([47.38, 67.97, 77.65, 82.46]),
        'std':  np.array([]),
      },
      {
        'label': 'iCaRL',
        'mean': np.array([49.21, 51.61, 50.38, 50.93]),
        'std':  np.array([]),
      },
    ]
  
  if dataset == 'CIFAR10':
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
        'label': 'CE upper-bound',
        'mean': np.array([70.38, 70.38, 70.38, 70.38]),
        'std':  np.array([0.57, 0.57, 0.57, 0.57]),
      },

      # Episodic approaches
      {
        'label': 'ML-PT',
        'mean': np.array([41.77, 52.50, 60.32, 66.45]),
        'std':  np.array([]),
      },
      {
        'label': 'ML-RP',
        'mean': np.array([39.00, 49.28, 58.92, 64.53]),
        'std':  np.array([]),
      },
      {
        'label': 'ML-CE',
        'mean': np.array([41.95, 53.18, 60.58, 65.80]),
        'std':  np.array([]),
      },
      # Metric approaches
      {
        'label': 'MT-NX',
        'mean': np.array([38.52, 45.00, 48.83, 56.35]),
        'std':  np.array([0.67, 2.33, 1.15, 2.23]),
      },      
      {
        'label': 'MT-CO',
        'mean': np.array([36.07, 45.52, 51.90, 59.15]),
        'std':  np.array([2.67, 2.47, 3.63, 1.27]),
      },
      {
        'label': 'MT-TR',
        'mean': np.array([33.50, 41.12, 46.50, 58.67]),
        'std':  np.array([2.35, 2.07, 3.10, 2.11]),
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
      {
        'label': 'GSS',
        'mean': np.array([28.48, 30.10, 31.29, 35.22]),
        'std':  np.array([]),
      },
      {
        'label': 'iCaRL',
        'mean': np.array([22.93, 21.03, 19.44, 25.64]),
        'std':  np.array([]),
      },
    ]


  return data

def main():
  datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
  colors = [
    'springgreen',
    'seagreen',
    'olivedrab',
  
    'steelblue',
    'deepskyblue',
    'darkturquoise',

    'darkorange',
    'goldenrod',
    'orange',

    'darkorchid',
    'slateblue',
    'mediumorchid',
    'red', 
    'salmon'
  ]
  
  fig, axs = plt.subplots(1, len(datasets), figsize=(15,6))
  axs[0].set_ylabel('Accuracy', fontsize=12)
  
  for ds_idx, dataset in enumerate(datasets):
    data = get_data(dataset)

    for idx, method in enumerate(data):
      label = method['label']
      mean = method['mean']
      std = method['std']
      axs[ds_idx].plot(np.arange(4), mean, '-o', color=colors[idx], label=label)
      # axs[ds_idx].fill_between(
      #   x, mean-std, mean+std,
      #   edgecolor=colors[idx],
      #   facecolor=colors[idx],
      #   alpha=0.2)
  
    axs[ds_idx].set_title(dataset, fontsize=14)
    axs[ds_idx].set_xlabel('Memory size', fontsize=12)
    axs[ds_idx].set_xticks(np.arange(4))
    axs[ds_idx].set_xticklabels(['0.2K', '0.5K', '1K', '2K'])
    if dataset == 'CIFAR10':
      axs[ds_idx].set_yticks(np.arange(15, 86, step=10)) # For CIFAR10
    else:
      axs[ds_idx].set_yticks(np.arange(45, 96, step=10)) # For FMNIST
    
    pos = axs[ds_idx].get_position()
    axs[ds_idx].set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
  
  axs[1].legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.3),
    ncol=5, 
  )
  # axs[1].legend(
  #   loc='lower center',
  #   bbox_to_anchor=(0, 1.02, 1, 0.2),
  #   ncol=5,
  #   fancybox=True,
  #   shadow=False)

  # handles, labels = axs[ds_idx].get_legend_handles_labels()
  # fig.legend(handles, labels, loc='upper center', ncol=5, borderaxespad=3)
  plt.show()

if __name__ == '__main__':
  main()
  