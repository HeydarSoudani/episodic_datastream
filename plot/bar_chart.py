import numpy as np
import matplotlib.pyplot as plt


def get_data(dataset):

  data = {
    'MNIST': {
      'mean': np.array([
        [86.80, 91.47, 92.67, 93.27], # epis-pt
        [86.63, 89.70, 91.07, 91.66], # epis-ce
        [43.93, 65.97, 79.00, 85.57], # non-epis
        [80.37, 86.80, 91.15, 93.00], # cope
        [86.70, 92.70, 94.30, 95.30], # mir
        [79.09, 86.78, 90.24, 91.59], # reservoir
        [71.87, 72.67, 73.18, 73.24], # icarl
      ]),
      'std':  np.array([
        [0.22, 0.68, 0.21, 0.09], # epis-pt
        [0.12, 0.29, 0.25, 0.12], # epis-ce
        [0.58, 1.01, 0.08, 0.34], # non-epis
        [1.87, 1.51, 0.72, 0.45], # cope
        [1.60, 0.40, 0.40, 0.40], # mir
        [1.35, 0.87, 0.80, 0.89], # reservoir
        [0.79, 0.50, 0.67, 0.51]  # icarl
      ]),
      'mem_avg': [
        '91.05±2.54', # epis-pt
        '89.77±1.94', # epis-ce
        '68.62±15.9', # non-epis
        '87.83±4.86', # cope
        '92.25±3.34', # mir
        '86.93±4.85', # reservoir
        '72.74±0.55', # icarl
      ],
    },
    'FashionMNIST': {
      'mean': np.array([
        [78.97, 84.87, 87.03, 88.63], # epis-pt
        [80.00, 84.77, 87.27, 89.13], # epis-ce
        [55.57, 70.30, 78.40, 83.13], # non-epis
        [75.92, 79.14, 81.84, 83.78], # cope
        [66.00, 74.80, 72.20, 75.50], # mir
        [72.61, 79.72, 83.63, 83.25], # reservoir
        [49.21, 51.61, 50.38, 50.93], # icarl
      ]),
      'std':  np.array([
        [1.27, 0.46, 0.45, 0.39], # epis-pt
        [1.23, 0.29, 0.21, 0.12], # epis-ce
        [2.37, 2.04, 0.41, 1.44], # non-epis
        [0.65, 4.99, 2.79, 2.20], # cope
        [4.00, 1.70, 5.00, 2.40], # mir
        [1.16, 3.01, 1.22, 0.54], # reservoir
        [1.74, 2.70, 2.10, 1.40]  # icarl
      ]),
      'mem_avg': [
        '84.88±3.66', # epis-pt
        '85.29±3.42', # epis-ce
        '71.85±10.46', # non-epis
        '80.17±2.96', # cope
        '72.13±3.74', # mir
        '79.80±4.42', # reservoir
        '50.53±0.88', # icarl
      ],
    },
    'CIFAR10': {
      'mean': np.array([
        [41.67, 50.70, 57.13, 61.43], # epis-pt
        [41.47, 51.07, 56.90, 61.07], # epis-ce
        [19.87, 33.53, 41.43, 49.60], # non-epis
        [29.47, 33.93, 31.84, 28.71], # cope
        [25.80, 32.30, 36.50, 45.20], # mir
        [37.06, 46.06, 46.75, 49.90], # reservoir
        [22.93, 21.03, 19.44, 25.64], # icarl
      ]),
      'std':  np.array([
        [0.25, 0.41, 2.02, 0.98], # epis-pt
        [0.95, 0.54, 0.75, 2.15], # epis-ce
        [0.76, 1.97, 2.70, 1.40], # non-epis
        [5.27, 1.75, 3.58, 5.80], # cope
        [1.00, 1.40, 1.30, 2.10], # mir
        [0.49, 6.20, 15.3, 5.80], # reservoir
        [2.12, 2.75, 2.86, 2.63]  # icarl
      ]),
      'mem_avg': [
        '52.73±7.44', # epis-pt
        '52.63±7.36', # epis-ce
        '36.11±10.96', # non-epis
        '30.99±2.05', # cope
        '34.95±7.04', # mir
        '44.94±4.78', # reservoir
        '22.26±2.31', # icarl
      ],
    },
    # {
    #   'dataset': 'cifar100',
    #   'mean': np.array([
    #     [], # epis-pt
    #     [], # epis-ce
    #     [], # non-epis
    #     [], # cope
    #     [], # mir
    #     [], # reservoir
    #     [], # icarl
    #   ]),
    #   'std':  np.array([
    #     [], # epis-pt
    #     [], # epis-ce
    #     [], # non-epis
    #     [], # cope
    #     [], # mir
    #     [], # reservoir
    #     []  # icarl
    #   ]),
    #   'mem_avg': [
    #     '±', # epis-pt
    #     '±', # epis-ce
    #     '±', # non-epis
    #     '±', # cope
    #     '±', # mir
    #     '±', # reservoir
    #     '±', # icarl
    #   ],
    }

  return data[dataset]



def main():
  dataset = 'FashionMNIST' #[MNIST, FashionMNIST, CIFAR10]
  data = get_data(dataset)
  methods = ['Epis-PT', 'Epis-CE', 'Non-Epis', 'CoPE', 'MIR', 'reservoir', 'iCaRL']
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen', 'brown']
  n_methods = 7
  n_mem = 4
  ind = np.arange(n_mem)  # the x locations for the groups
  width = 0.1
  
  mean = data['mean']
  std = data['std']
  mem_avg = data['mem_avg']
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  for i in range(n_methods):
    index = ind + (i-3.0)*width 
    ax.bar(
      index,
      mean[i],
      yerr=std[i],
      color=colors[i],
      width=width,
      align='center',
      label='{}: {}'.format(methods[i], mem_avg[i])
    )

  ax.set_xticks(ind)
  ax.set_xticklabels(['0.2K', '0.5K', '1K', '2K'], fontsize=16)
  ax.set_xlabel('Memory size', fontsize=18)
  ax.set_ylabel('Accuracy (%)', fontsize=18)
  ax.set_title('Split-{}'.format(dataset), fontsize=18, pad=62.0)
  ax.legend(loc='center', bbox_to_anchor=(0.5, 1.06),
          fancybox=True, shadow=False, ncol=4, fontsize=16)

  if dataset == 'MNIST':
    ax.set_ylim([40, 100])
    ax.set_yticklabels(np.arange(40, 101, step=10), fontsize=16)
  elif dataset == 'FashionMNIST':
    ax.set_ylim([40, 95])
    ax.set_yticklabels(np.arange(40, 100, step=10), fontsize=16)
  elif dataset == 'CIFAR10':
    ax.set_ylim([0, 70])
    ax.set_yticklabels(np.arange(0, 71, step=10), fontsize=16)

  fig.savefig('bar.png', format='png', dpi=2000)
  plt.show()

if __name__ == '__main__':
  main()