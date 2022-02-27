import numpy as np
import matplotlib.pyplot as plt


def main():
  dataset = 'cifar100' #['fmnist', 'cifar10', 'cifar100']
  class_num = 10
  colors = [

  ]
  data_points = np.array([5530, 12009, 17163, 21334, 26423, 32294, 40472, 49591, 61954, 63999])
  known_acc   = np.array([0.8161, 0.9082, 0.8276, 0.8271, 0.8182, 0.8158, 0.8050, 0.8396, 0.8558, 0.8709])
  # plt.plot(data_points, known_acc, '-o') #color=colors[idx], label=label

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

  # for i in range(class_num):
  #   plt.plot(data_points, known_acc_by_class[i], '-o', label='class {}'.format(i))
  #   plt.axvline(x=data_points[i], linestyle='--') #color='k'
  # plt.legend(loc='lower right')
  # plt.xlabel('Stream data')
  # plt.ylabel('Known class Accuracy')
  # plt.xticks(np.arange(0, 64000, step=10000))
  # plt.yticks(np.arange(0.4, 1.1, step=0.1)) 
  # plt.show()
  
  fig, axs = plt.subplots(class_num, 1)
  for class_idx, class_acc in enumerate(known_acc_by_class):
    axs[class_idx].plot(data_points, class_acc)
    plt.subplots_adjust(hspace = .001)
    if class_idx > 4:
      axs[class_idx].axvline(x=start_points[class_idx-5], linestyle='--') #color='k'
      axs[class_idx].axvline(x=detected_points[class_idx-5], linestyle='--')
    # axs[class_idx].set_title('Class {}'.format(class_idx))
    # axs[class_idx].set_xlabel('Stream data')
    
    axs[class_idx].set_ylabel('Label {}'.format(class_idx), rotation=0, ha='right')
    axs[class_idx].set_yticks(np.arange(0.6, 1.05, step=0.2)) 
    axs[class_idx].set_xticks(np.arange(0, 64000, step=10000))
    if class_idx != 9:
      axs[class_idx].set_xticklabels(())
  
  # plt.xlabel('Stream data')
  # plt.ylabel('Known class Accuracy')
  plt.show()





if __name__ == '__main__':
  main()
