import numpy as np
import matplotlib.pyplot as plt

def get_data():
  # cpe_1, cpe_3, cpe_5,
  # MT_ce, MT_xn, MT_co, MT_tr,
  data = [
    {
      'dataset': 'FashionMNIST',
      'mean': np.array([
        2013.15, 3615.75, 5669.10,
        # 280.77,
        278.57, 266.51, 269.08,
        568.29, 572.70,
      ]),
      'std': np.array([
        257.01, 419.60, 268.74,
        # 2.84,
        4.81, 4.12, 1.35,
        2.58, 9.43,
      ])
    },
    {
      'dataset': 'CIFAR10',
      'mean': np.array([

      ]),
      'std': np.array([

      ])
    }
  ]

  return data

def main():
  data = get_data()
  methods = ['CPE_e1', 'CPE_e3', 'CPE_e5', 'MT_XN', 'MT_CO', 'MT_TR']  #'MT_CE',
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen', 'brown']
  n_methods = 6
  n_set = 3
  ind = np.arange(n_set)
  width = 0.1
  
  fig, axs = plt.subplots(nrows=1, ncols=len(data), constrained_layout=True, figsize=(8,8))
  for idx, item in enumerate(data):
    dataset = item['dataset']
    mean = item['mean']
    std = item['std']
    for i in range(n_methods):
      index = ind + (i-2.5)*width 
      axs[idx].bar(
        index,
        mean[i],
        yerr=std[i],
        color=colors[i],
        width=width,
        align='center',
        label='{}'.format(methods[i])
    
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(['CPE', 'Metric', 'MetaLearning'], fontsize=8)
    axs[idx].set_xlabel('Methods', fontsize=12)
    axs[idx].set_ylabel('Run time (s)', fontsize=12)
    axs[idx].set_title('{}'.format(dataset), fontsize=14, pad=50.0)
    axs[idx].legend(loc='center', bbox_to_anchor=(0.5, 1.12),
      fancybox=True, shadow=False, ncol=3, fontsize=11.5)

  # fig.savefig('bars.png', format='png', dpi=1400)
  plt.show()



