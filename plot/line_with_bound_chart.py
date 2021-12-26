import numpy as np
import matplotlib.pyplot as plt

# [86.48, 72.86, 73.23, 65.18, 63.38], # epis-pt b=0.999 (dist)

def get_data(dataset):
  # methods on rows
  # tasks on cols.

  if dataset == 'pmnist':
    data = [
      {
        'label': 'Without memory',
        'mean': np.array([
          [93.36, 91.35, 85.82, 78.36, 74.25], # epis-rp
          [92.98, 90.76, 87.88, 81.32, 73.90], # epis-pt
          [89.68, 87.48, 83.78, 79.07, 74.15], # epis-ce
          [91.84, 87.26, 81.76, 74.92, 68.43], # non-epis
        ]),
        'std':  np.array([
          [00.11, 00.75, 01.23, 02.97, 05.92],
          [00.13, 00.40, 01.34, 05.10, 06.33],
          [00.28, 00.21, 01.78, 02.77, 02.43],
          [00.10, 00.93, 01.97, 03.15, 03.44],
        ])
      },
      {
        'label': 'Memory: 5 samples per class',
        'mean': np.array([
          [93.37, 91.55, 90.17, 86.14, 85.16],
          [92.50, 90.36, 87.91, 85.59, 83.11],
          [89.72, 87.15, 85.39, 82.53, 77.99],
          [89.63, 85.38, 80.08, 79.75, 76.75],
        ]),
        'std': np.array([
          [00.10, 00.40, 00.40, 00.50, 00.89],
          [00.07, 00.46, 00.86, 01.21, 02.16],
          [00.20, 00.28, 00.45, 02.16, 02.49],
          [00.23, 00.88, 02.90, 01.69, 01.95],
        ])
      },
      {
        'label': 'Memory: 10 sample per class',
        'mean': np.array([
          [93.35, 91.33, 89.74, 89.02, 86.83], # epis-rp
          [92.46, 90.79, 89.20, 86.57, 84.67],  # epis-pt
          [89.67, 86.91, 84.11, 82.12, 82.23], # epis-ce
          [91.19, 87.23, 84.82, 83.24, 79.85], # non-epis
        ]),
        'std': np.array([
          [00.05, 00.17, 00.57, 00.46, 00.36],
          [00.16, 00.47, 00.28, 01.54, 01.16],
          [00.25, 01.12, 01.33, 02.58, 01.60],
          [00.90, 01.17, 00.87, 01.82, 00.79],
        ])
      }
    ]

  elif dataset == 'rmnist':
    data = [
      {
        'label': 'Without memory',
        'mean': np.array([
          [93.35, 93.36, 89.09, 78.89, 64.69],
          [92.48, 92.47, 87.73, 77.78, 64.38],
          [89.69, 89.59, 85.45, 76.36, 64.83],
          [89.63, 89.25, 84.81, 75.38, 62.54] # non-epis
        ]),
        'std': np.array([
          [00.05, 00.04, 00.65, 00.84, 00.88],
          [00.13, 00.10, 00.25, 00.13, 00.36],
          [00.24, 00.05, 00.21, 00.08, 01.05],
          [00.23, 00.04, 00.08, 00.65, 01.47]
        ])
      },
      {
        'label': 'Memory: 5 samples per class',
        'mean': np.array([
          [93.43, 93.28, 89.10, 79.80, 66.66],
          [92.58, 92.55, 88.15, 78.91, 66.19],
          [89.59, 89.59, 85.42, 76.36, 65.51],
          [89.63, 89.26, 84.80, 75.60, 63.04] # non-epis
        ]),
        'std': np.array([
          [00.08, 00.07, 00.33, 00.59, 00.82],
          [00.21, 00.09, 00.28, 00.20, 00.32],
          [00.26, 00.12, 00.15, 00.19, 00.46],
          [00.23, 00.02, 00.21, 00.58, 01.43]
        ])
      },
      {
        'label': 'Memory: 10 samples per class',
        'mean': np.array([
          [93.39, 93.31, 89.22, 80.04, 67.01],
          [92.44, 92.46, 88.05, 78.89, 66.83],
          [89.67, 89.59, 85.41, 76.87, 65.61], # epis-ce
          [89.63, 89.33, 84.79, 75.35, 63.20] # non-epis
        ]),
        'std': np.array([
          [00.07, 00.06, 00.46, 00.51, 00.55],
          [00.16, 00.03, 00.15, 00.62, 01.39],
          [00.24, 00.08, 00.14, 00.31, 00.51],
          [00.23, 00.07, 00.20, 00.12, 01.35]
        ])
      },
      {
        'label': 'Memory: 50 samples per class',
        'mean': np.array([
          [93.39, 93.29, 89.71, 83.21, 75.26],
          [92.45, 92.57, 88.67, 81.86, 73.63],
          [89.67, 89.63, 85.67, 78.18, 69.63],
          [89.63, 89.42, 85.12, 77.12, 67.41] # non-epis
        ]),
        'std': np.array([
          [00.09, 00.06, 00.40, 00.35, 00.65],
          [00.11, 00.09, 00.23, 00.15, 00.75],
          [00.22, 00.11, 00.21, 00.09, 01.04],
          [00.23, 00.05, 00.11, 00.19, 01.39]
        ])
      }
    ]

  elif dataset == 'pfmnist':
    data = [
      {
        'label': 'Without memory',
        'mean': np.array([
          [], [91.75, 74.60, 72.70, 65.00, 57.18], # reptile
          [85.66, 77.19, 71.54, 72.71, 64.66], #[85.42, 73.82, 72.99, 66.86, 67.21], # epis-pt b=1.0
          [88.43, 78.70, 73.96, 73.31, 64.60], #[88.25, 74.87, 75.36, 66.33, 62.57], # epis-ce
          [90.98, 77.78, 70.89, 70.37, 57.79], #[91.26, 77.77, 70.90, 67.28, 56.54], # non-epis

          [85.62, 72.86, 75.34, 67.29, 65.19], # epis-pt b=0.999
          [85.16, 72.27, 73.59, 65.47, 62.45], # epis-pt b=0.9
        ]),
        'std': np.array([
          [], [00.20, 02.81, 09.16, 07.71, 11.98],
          [00.55, 03.95, 06.58, 08.12, 09.53], #[00.61, 01.91, 08.38, 06.55, 03.15],
          [00.33, 04.31, 05.92, 09.83, 12.86], #[00.29, 01.85, 05.70, 08.44, 08.37],
          [00.52, 03.27, 03.38, 06.24, 11.67], #[00.22, 00.47, 04.12, 02.83, 04.16],

          [00.29, 01.87, 02.64, 06.12, 05.11],
          [00.86, 02.61, 07.77, 07.85, 08.43], 
        ]),
      },
      {
        'label': 'Memory: 10 samples per class',
        'mean': np.array([
          [91.43, 84.62, 83.26, 81.76, 78.42], # reptile
          [86.71, 82.28, 81.10, 80.05, 80.69], # epis-pt b=1.0
          [88.00, 81.45, 81.76, 76.14, 77.63], # epis-ce
          [90.95, 84.36, 82.12, 79.62, 77.49], # non-epis

          [], # epis-pt b=0.999
          [], # epis-pt b=0.9
        ]),
        'std': np.array([
          [00.76, 02.56, 01.52, 04.38, 03.81],
          [01.31, 03.24, 02.95, 04.94, 04.21],
          [00.26, 02.70, 01.71, 09.38, 05.88],
          [00.41, 02.18, 02.63, 04.66, 05.61], 

          [],
          [],
        ]),
      },
    ]

  return data

def main():
  dataset = 'pfmnist' #['pmnist', 'rmnist', 'pfmnist]
  # methods = ['epis-rp', 'epis-pt', 'epis-ce', 'non-epis']
  methods = ['epis-pt:beta0.999', 'non-epis']
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen']

  data = get_data(dataset)
  n_task = 5
  x = np.arange(n_task)

  fig, axs = plt.subplots(1, len(data))
  # for idx, item in enumerate(data):
  #   label = item['label']
  #   mean = item['mean']
  #   std = item['std']
    # for i in range(len(methods)):
    #   axs[idx].plot(
    #     x, mean[i],
    #     '-o', color=colors[i],
    #     label=methods[i])
    #   axs[idx].fill_between(
    #     x, mean[i]-std[i], mean[i]+std[i],
    #     edgecolor=colors[i], facecolor=colors[i],
    #     alpha=0.2)

    # axs[idx].legend(loc='lower left')
    # axs[idx].set_title(label)
    # axs[idx].set_xlabel('tasks')
    # axs[idx].set_ylabel('Accuracy')
    # axs[idx].set_xticks(np.arange(0, 4.1, step=1.))
    # axs[idx].set_yticks(np.arange(60, 96, step=5))



  label = data[0]['label']
  mean = data[0]['mean']
  std = data[0]['std']
  for i in range(len(methods)):
    plt.plot(
      x, mean[i],
      '-o', color=colors[i],
      label=methods[i])
    plt.fill_between(
      x, mean[i]-std[i], mean[i]+std[i],
      edgecolor=colors[i], facecolor=colors[i],
      alpha=0.2)

  plt.legend(loc='lower left')
  plt.title(label)
  plt.xlabel('tasks')
  plt.ylabel('Accuracy')
  plt.xticks(np.arange(0, 4.1, step=1.))
  plt.yticks(np.arange(40, 96, step=5))


  # fig.savefig('cca.png', format='png', dpi=800)
  plt.show()

if __name__ == '__main__':
  main()

