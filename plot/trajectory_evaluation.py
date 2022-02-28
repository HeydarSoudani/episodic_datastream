import numpy as np
import matplotlib.pyplot as plt


def avg_known_classes():

  data_points = np.array([5530, 12009, 17163, 21334, 26423, 32294, 40472, 49591, 61954, 63999])
  known_acc   = np.array([0.8161, 0.9082, 0.8276, 0.8271, 0.8182, 0.8158, 0.8050, 0.8396, 0.8558, 0.8709])
  mnew = np.array([])
  plt.plot(data_points, known_acc, '-o') #color=colors[idx], label=label


def main():
  dataset = 'MNIST' #['fmnist', 'cifar10', 'cifar100']
  class_num = 10
  colors = ['limegreen', 'hotpink', 'blueviolet', 'royalblue', 'darkorange', 'gold', 'brown']

  known_acc_by_class = np.array([
    [0.8082, 0.9116, 0.8816, 0.8739, 0.8430, 0.8459, 0.8318, 0.8388, 0.8271, 0.8717],
    [0.8809, 0.9087, 0.9636, 0.9504, 0.9399, 0.9253, 0.9494, 0.9222, 0.9184, 0.9424],
    [0.7879, 0.8936, 0.8484, 0.8362, 0.8432, 0.7783, 0.7915, 0.7871, 0.7994, 0.8333],
    [0.7869, 0.9087, 0.8304, 0.7983, 0.7775, 0.7607, 0.7675, 0.8074, 0.7709, 0.7967],
    [0.8128, 0.9182, 0.8863, 0.8514, 0.8687, 0.8222, 0.7529, 0.7485, 0.7405, 0.7563],
    [None, None, 0.6024, 0.7054, 0.7441, 0.7683, 0.7031, 0.7493, 0.7847, 0.7561],
    [None, None, None, 0.8133, 0.8664, 0.9090, 0.9334, 0.9369, 0.9450, 0.9530],
    [None, None, None, None, 0.7386, 0.8422, 0.7760, 0.7643, 0.8305, 0.7296],
    [None, None, None, None, None, 0.7233, 0.8443, 0.8703, 0.8522, 0.9255],
    [None, None, None, None, None, None, 0.7418, 0.8717, 0.9180, 0.9565],
  ]).astype(np.double)

  start_points = np.array([6005, 12011, 18002, 24001, 30001])
  detected_points = np.array([12009, 17163, 21334, 26423, 32294])

  # plt.rcParams['axes.grid'] = True
  fig, axs = plt.subplots(6, 1)
  fig.subplots_adjust(hspace = .001)
  # plt.grid()
  plt.suptitle('{} dataset'.format(dataset))

  for i in range(5):
    axs[0].plot(data_points, known_acc_by_class[i], '-o', label='class {}'.format(i))
  axs[0].set_ylim([0.5, 1])
  axs[0].set_ylabel('Base classes', fontsize=12, rotation=0, ha='right')
  axs[0].set_yticks(np.arange(0.6, 1.05, step=0.2))
  axs[0].set_xticks(np.arange(0, 64000, step=10000))
  axs[0].legend(loc='lower right', ncol=5)

  for class_idx, class_acc in enumerate(known_acc_by_class[5:]):    
    axs[class_idx+1].plot(data_points, class_acc, '-o', color=colors[class_idx])
    axs[class_idx+1].axvline(x=start_points[class_idx], linestyle='--', color=colors[class_idx]) #color='k'
    axs[class_idx+1].axvline(x=detected_points[class_idx], linestyle='-.', color=colors[class_idx])
    axs[class_idx+1].axvspan(start_points[class_idx], detected_points[class_idx], alpha=0.25, color=colors[class_idx])
    # axs[class_idx+1].grid()
    axs[class_idx+1].set_ylim([0.5, 1])
    axs[class_idx+1].set_ylabel('Label {}'.format(class_idx+5), fontsize=12, rotation=0, ha='right')
    axs[class_idx+1].set_yticks(np.arange(0.6, 1.05, step=0.2)) 
    axs[class_idx+1].set_xticks(np.arange(0, 64000, step=5000))

    axs[class_idx+1].annotate(
      "",
      xy=(start_points[class_idx], 0.75), xycoords='data',
      xytext=(detected_points[class_idx], 0.75), textcoords='data',
      arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color=colors[class_idx], lw=1),
    )
    axs[class_idx+1].text(
      int(0.5*(detected_points[class_idx] + start_points[class_idx]) - 730 ), 0.8,
      '%g'%(detected_points[class_idx] - start_points[class_idx]), 
      rotation=0, fontsize=10, color=colors[class_idx])

    if class_idx != 4:
      axs[class_idx+1].set_xticklabels(())
  
  plt.xlabel('Stream data')
  # plt.ylabel('Known class Accuracy')
  plt.show()


if __name__ == '__main__':
  # main()
  avg_known_classes()
