import numpy as np
import matplotlib.pyplot as plt
from math import floor

def get_stream_data():
  # cpe_1, cpe_3, cpe_5,
  # MT_xn, MT_co, MT_tr,
  # EP_ce, EP_pt, EP_rp
  data = [
    {
      'dataset': 'FashionMNIST',
      'mean': np.array([
        2013.15, 3615.75, 5669.10,
        278.57, 266.51, 269.08,
        568.29, 572.70, 5408.25
      ]),
      'std': np.array([
        257.01, 419.60, 268.74,
        4.81, 4.12, 1.35,
        2.58, 9.43, 150.14
      ])
    },
    {
      'dataset': 'CIFAR10',
      'mean': np.array([
        1181.12, 2286.18, 3156.93,
        272.52, 254.56, 267.68,
        1138.71, 1087.89, 4022.36
      ]),
      'std': np.array([
        156.81, 453.53, 537.65,
        6.19, 6.34, 2.78,
        33.43, 39.26, 156.88
      ])
    }
  ]
  return data

def get_details_stream_data():
  # 'retrain', 'detector', 'memory', 'evaluation'
  data = [
    {
      'dataset':'FashionMNIST',
      'mean': np.array([
        [41.20, 57.40, 0.29, 185.62], # MT_xn
        [28.73, 52.77, 0.27, 184.74], # MT_co
        [29.74, 54.47, 0.28, 184.59], # MT_tr
        [324.59, 51.46, 0.28, 191.96], # EP_ce
        [329.37, 50.94, 0.28, 192.10], # EP_pt
        [5133.64, 83.41, 0.43, 190.77], # EP_rp
      ]),
      'std':  np.array([
        [3.53, 7.55, 0.02, 0.53], # MT_xn
        [1.38, 2.54, 0.02, 0.76], # MT_co
        [0.49, 0.74, 0.02, 0.95], # MT_tr
        [2.24, 0.97, 0.01, 0.84], # EP_ce
        [8.42, 0.61, 0.01, 1.40], # EP_pt
        [148.65, 1.64, 0.03, 0.14], # EP_rp
      ]),
    },
    {
      'dataset': 'CIFAR10',
      'mean': np.array([
        [50.98, 62.09, 0.36, 159.09], # MT_xn
        [37.32, 59.58, 0.36, 157.31], # MT_co
        [41.49, 66.65, 0.42, 159.12], # MT_tr
        [902.03, 71.87, 0.55, 164.26], # EP_ce
        [855.33, 68.38, 0.51, 163.67], # EP_pt
        [3802.00, 55.53, 0.42, 164.41], # EP_rp
      ]),
      'std':  np.array([
        [2.18, 3.30, 0.00, 0.89], # MT_xn
        [2.43, 3.55, 0.02, 0.74], # MT_co
        [1.40, 2.10, 0.02, 1.18], # MT_tr
        [30.90, 1.34, 0.04, 1.59], # EP_ce
        [36.02, 2.79, 0.02, 0.73], # EP_pt
        [152.72, 2.00, 0.01, 3.79], # EP_rp
      ]),
    }
  ]

  return data


# def get_incremental_data():
#   # iCaRL, GSS, MIR, Res, CoPE,
#   # MT_xn, MT_co, MT_tr,
#   # EP_ce, EP_pt, EP_rp
#   data = [
#     {
#       'dataset': 'MNIST',
#       'mean': np.array([
#         , , , , 119.26,
#         290.50, 63.26, 62.38,
#         578.39, 584.54, 1094.08
#       ]),
#       'std': np.array([
#         , , , , 1.31,
#         2.69, 1.25, 0.33,
#         1.67, 8.48, 10.71
#       ])
#     },
#     {
#       'dataset': 'FashionMNIST',
#       'mean': np.array([
#         , , , , 496.31,
#         , , ,
#         , , ,
#       ]),
#       'std': np.array([
#         , , , , 1.16,
#         , , ,
#         , , ,
#       ])
#     },
#     {
#       'dataset': 'CIFAR10',
#       'mean': np.array([
#         , , , , 334.01,
#         , , ,
#         2394.98, 2407.66, 6073.16,
#       ]),
#       'std': np.array([
#         , , , , 2.18,
#         , , ,
#         11.78, 30.13, 27.34,
#       ])
#     }
#   ]

#   return data

def all_time_plot():
  data = get_stream_data()
  methods = ['CPE_e1', 'CPE_e3', 'CPE_e5', 'XN_PE', 'CO_PE', 'TR_PE', 'MPE_CE', 'MPE_PT', 'MPE_RP' ]  #'MT_CE',
  colors = [
    'orchid', 'darkorchid', 'blueviolet',
    'darkorange', 'goldenrod', 'peru',
    'darkgreen', 'olivedrab', 'yellowgreen',
    ]
  n_methods = 9
  n_set = 3
  ind = np.arange(n_set)
  width = 0.14
  
  fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
  # fig.subplots_adjust(hspace = .001)
  
  for idx, item in enumerate(data):
    dataset = item['dataset']
    mean = item['mean']
    std = item['std']
    for i in range(n_methods):
      index = 0.6*floor(i/n_set) + ((i%n_set)-1.5)*width 
      axs[idx].bar(
        index,
        mean[i],
        yerr=std[i],
        color=colors[i],
        width=width,
        # align='center',
        align='edge',
        label='{}'.format(methods[i])
      )
    
    axs[idx].set_ylim([0, 6000])
    axs[idx].set_xlim([-0.4, 1.6])
    axs[idx].set_xticks([0, 0.6, 1.2])
    axs[idx].set_xticklabels(['CPE', 'Metric', 'MetaLearning'], fontsize=12)
    axs[idx].set_yticklabels(np.arange(0, 6001, step=1000), fontsize=10, rotation=45)
    # axs[idx].set_xlabel('Methods', fontsize=10)
    axs[idx].set_ylabel('Run time (sec)', fontsize=12)
    axs[idx].set_title('{}'.format(dataset), fontsize=12)

    # if idx == 0:
    pos = axs[idx].get_position()
    axs[idx].set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
  
  handles, labels = axs[idx].get_legend_handles_labels()
  fig.legend(handles, labels, ncol=3, loc='upper center', fontsize=12)
  fig.legend(loc='center', bbox_to_anchor=(0.5, 1.12),
    fancybox=True, shadow=False, ncol=3, fontsize=11.5)

  # fig.savefig('bars.png', format='png', dpi=1400)
  plt.show()

def details_times_plot():
  methods = ['MT_XN', 'MT_CO', 'MT_TR', 'EP_CE', 'EP_PT', 'EP_RP' ]  #'MT_CE',
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen']
  n_methods = 6
  width = 0.1
  ind = np.arange(4)
  data = get_details_stream_data()

  fig, axs = plt.subplots(nrows=1, ncols=len(data), figsize=(11, 4.5))
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
      )

    axs[idx].set_yscale('log')  
    axs[idx].set_xticks(ind)
    axs[idx].set_xticklabels(['retrain', 'detector', 'memory', 'evaluation'], fontsize=12)
    axs[idx].set_xlabel('Component', fontsize=14)
    axs[idx].set_ylabel('Run time (s)', fontsize=14)
    axs[idx].set_title('{}'.format(dataset), fontsize=16)

    pos = axs[idx].get_position()
    axs[idx].set_position([pos.x0, pos.y0, pos.width, pos.height * 0.82])
  
  handles, labels = axs[idx].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)

  # fig.savefig('bars.png', format='png', dpi=1400)
  plt.show()


if __name__ == '__main__':
  all_time_plot()
  # details_times_plot()

