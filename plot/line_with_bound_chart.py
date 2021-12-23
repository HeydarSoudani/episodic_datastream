import numpy as np
import matplotlib.pyplot as plt


def get_data(dataset):
  # methods on rows
  # tasks on cols.

  if dataset == 'pmnist':
    # no-mem
    # mean = np.array([
    #   [91.84, 87.26, 81.76, 74.92, 68.43], # non-epis
    #   [89.68, 87.48, 83.78, 79.07, 74.15], # epis-ce
    #   [93.36, 91.35, 85.82, 78.36, 74.25], # epis-rp
    #   [92.98, 90.76, 87.88, 81.32, 73.90], # epis-pt
    # ])
    # std = np.array([
    #   [00.10, 00.93, 01.97, 03.15, 03.44],
    #   [00.28, 00.21, 01.78, 02.77, 02.43],
    #   [00.11, 00.75, 01.23, 02.97, 05.92],
    #   [00.13, 00.40, 01.34, 05.10, 06.33],
    # ])

    # mem: 5pc
    mean = np.array([
      [89.63, 85.38, 80.08, 79.75, 76.75],
      [89.72, 87.15, 85.39, 82.53, 77.99],
      [93.37, 91.55, 90.17, 86.14, 85.16],
      [92.50, 90.36, 87.91, 85.59, 83.11]
    ])
    std = np.array([
      [00.23, 00.88, 02.90, 01.69, 01.95],
      [00.20, 00.28, 00.45, 02.16, 02.49],
      [00.10, 00.40, 00.40, 00.50, 00.89],
      [00.07, 00.46, 00.86, 01.21, 02.16]
    ])

    # mem: 10pc


  if dataset == 'rmnist':
    pass
  
  return mean, std

def main():
  dataset = 'pmnist' #['pmnist', 'rmnist']
  methods = ['non-epis', 'epis-ce', 'epis-rp', 'epis-pt']
  colors = ['royalblue', 'hotpink', 'blueviolet', 'gold', 'darkorange', 'limegreen']
  mean, std = get_data(dataset)
  n_task = 5
  x = np.arange(n_task)

  # fig = plt.figure(figsize=(8,6))
  for i in range(4):
    plt.plot(
      x, mean[i],
      '-o', color=colors[i],
      label=methods[i])
    plt.fill_between(
      x, mean[i]-std[i], mean[i]+std[i],
      edgecolor=colors[i], facecolor=colors[i],
      alpha=0.3)

  plt.legend(loc='best')
  plt.xlabel('tasks')
  plt.ylabel('Accuracy')
  plt.xticks(np.arange(0, 4.1, step=1.))
  # plt.yticks(np.arange(0.5, 1.01, step=0.1))

  # fig.savefig('cca.png', format='png', dpi=800)
  plt.show()

if __name__ == '__main__':
  main()

