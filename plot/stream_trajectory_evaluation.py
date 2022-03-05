import numpy as np
import matplotlib.pyplot as plt

def get_data():

  ### ==================================
  ## == Episodic-PT: 2k_iter, 10k_KB
  # data_points = np.array([5119, 9999, 11979, 19999, 21598, 29999, 34065, 39999, 49999, 50782, 59999, 63999])
  # start_points = np.array([3001, 8001, 10012, 16000, 26014])
  # detected_points = np.array([5119, 11979, 11979, 21598, 34065])
  # known_acc_by_class = np.array([
  #   [0.8351, 0.9056, 0.8718, 0.9278, 0.8783, 0.9276, 0.9339, 0.9549, 0.9272, None,   None,   None],
  #   [0.8878, 0.9536, 0.9184, 0.9555, 0.9430, 0.9500, 0.9239, 0.9695, 0.9474, 0.8860, 0.9522, None],
  #   [0.8150, 0.8816, 0.8731, 0.8977, 0.9024, 0.8838, 0.8473, 0.8994, 0.8721, None,   None,   None],
  #   [0.8496, 0.8865, 0.8865, 0.8920, 0.8888, 0.8841, 0.8953, 0.8956, 0.8501, None,   None,   None],
  #   [0.8331, 0.9154, 0.8581, 0.9083, 0.8769, 0.9219, 0.9266, 0.9159, 0.9056, None,   None,   None],
  #   [None,   0.7490, 0.7957, 0.8305, 0.8109, 0.7965, 0.7884, 0.8669, 0.8617, 0.7908, 0.8994, None],
  #   [None,   None,   None,   0.7882, 0.7500, 0.8426, 0.8281, 0.8972, 0.8954, 0.8703, 0.9479, None],
  #   [None,   None,   None,   0.8171, 0.7440, 0.8997, 0.8705, 0.9138, 0.8951, 0.8874, 0.9119, 0.9003],
  #   [None,   None,   None,   None,   None,   0.7975, 0.8517, 0.8814, 0.8479, None,   0.8871, 0.8762],
  #   [None,   None,   None,   None,   None,   None,   None,   0.7805, 0.7978, 0.8176, 0.8650, 0.9060] 
  # ]).astype(np.double)

  # CwCA_avg = np.array([0.8428, 0.8896, 0.8738, 0.8775, 0.8524, 0.8798, 0.8715, 0.8994, 0.8750, 0.8506, 0.9045, 0.8955])
  # Mnew_avg = np.array([0.3019, 0.4029, 0.3287, 0.7380, 0.6256, 0.8427, 0.6294, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
  # Fnew_avg = np.array([0.1499, 0.0718, 0.1159, 0.0831, 0.1203, 0.0557, 0.0911, 0.0384, 0.0690, 0.1047, 0.0345, 0.0450])

  ### ==================================
  ## == Metric-triplet: 25_epoch, 10k_KB
  data_points = np.array([4682, 9999, 11336, 16030, 19999, 20566, 27009, 29999, 33204, 39562, 39999, 45693, 49999, 53703, 59999, 63448, 63999])
  start_points = np.array([3001, 8001, 10012, 16000, 26014])
  detected_points = np.array([4682, 11336, 16030, 20566, 33204])
  known_acc_by_class = np.array([
    [0.8063, 0.8356, 0.8031, 0.8517, 0.8306, None,   0.8108, 0.8632, 0.8512, 0.8452, 0.8863, 0.8336, 0.8437, None,   None,   None,   None  ],
    [0.8733, 0.9177, 0.9351, 0.9417, 0.9569, 0.8787, 0.9208, 0.9337, 0.9049, 0.9463, 0.9387, 0.9349, 0.9395, 0.9050, 0.9126, None,   None  ], 
    [0.7872, 0.8218, 0.8315, 0.7826, 0.7959, 0.7625, 0.7810, 0.8333, 0.7777, 0.7436, 0.8043, 0.7662, 0.7701, None,   None,   None,   None  ],
    [0.8189, 0.8126, 0.8502, 0.8318, 0.7513, 0.7230, 0.7736, 0.7771, 0.7960, 0.8039, 0.8604, 0.7592, 0.7582, None,   None,   None,   None  ], 
    [0.8150, 0.8648, 0.8190, 0.8413, 0.8672, 0.8000, 0.8422, 0.8768, 0.8348, 0.8018, None,   0.7770, 0.7818, None,   None,   None,   None  ], 
    [None,   0.7238, 0.7323, 0.8000, 0.7931, 0.7922, 0.7566, 0.7530, 0.7459, 0.7222, 0.8163, 0.7123, 0.8183, 0.7925, 0.8366, None,   None  ],
    [None,   None,   None,   0.8228, 0.8494, 0.8307, 0.8605, 0.8798, 0.8563, 0.8618, 0.8666, 0.8378, 0.8747, 0.8504, 0.8454, None,   None  ],
    [None,   None,   None,   None,   0.7520, 0.6716, 0.7610, 0.8198, 0.8478, 0.7067, 0.7321, 0.7515, 0.7979, 0.7948, 0.7934, 0.7974, None  ],
    [None,   None,   None,   None,   None,   None,   0.7509, 0.8151, 0.7994, 0.8589, 0.8750, 0.8368, 0.9237, 0.8357, 0.9072, 0.8652, 0.8913],
    [None,   None,   None,   None,   None,   None,   None,   None,   None,   0.7142, 0.9400, 0.8312, 0.8736, 0.8487, 0.9323, 0.9332, 0.9903]
  ])

  CwCA_avg = np.array([0.8180, 0.8336, 0.8323, 0.8398, 0.8241, 0.7798, 0.8117, 0.8401, 0.8223, 0.8031, 0.8558, 0.8031, 0.8609, 0.8386, 0.8696, 0.8745, 0.9655])
  Mnew_avg = np.array([0.3333, 0.6471, 0.3618, 0.3290, 0.3313, 0.2500, 0.5446, 0.6222, 0.6760, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
  Fnew_avg = np.array([0.1785, 0.1123, 0.1503, 0.1337, 0.1491, 0.1960, 0.1499, 0.1277, 0.1567, 0.1573, 0.1190, 0.1665, 0.1115, 0.1404, 0.1015, 0.1047, 0.0145])

  return data_points, start_points, detected_points, known_acc_by_class, CwCA_avg, Mnew_avg, Fnew_avg

def UDA_plot():
  data_points, _, _, _, _, Mnew_avg, Fnew_avg = get_data()
  
  plt.plot(data_points, Mnew_avg, '-o', label='M_new')
  plt.plot(data_points, Fnew_avg, '-o', label='F_new')

  # plt.set_ylabel('Base classes', fontsize=12, rotation=0, ha='right')
  plt.title('Unknown Detection Accuracy')
  plt.yticks(np.arange(0.0, 1.05, step=0.2))
  plt.xticks(np.arange(0, 64000, step=10000))
  plt.legend(loc='best')
  plt.xlabel('Stream data')

  plt.show()

def avg_class_plot():
  dataset = 'MNIST' #['fmnist', 'cifar10', 'cifar100']
  class_num = 10
  colors = ['limegreen', 'hotpink', 'blueviolet', 'royalblue', 'darkorange', 'gold', 'brown']

  ### Get data
  data_points, start_points, detected_points, known_acc_by_class, CwCA_avg, _, _ = get_data()

  # plt.rcParams['axes.grid'] = True
  fig, axs = plt.subplots(7, 1)
  fig.subplots_adjust(hspace = .001)
  plt.suptitle('{} dataset'.format(dataset))

  axs[0].plot(data_points, CwCA_avg, '-o', label='CwCA', color=colors[-1])
  axs[0].set_ylim([0.5, 1])
  axs[0].set_ylabel('Average Acc.', fontsize=12, rotation=0, ha='right')
  axs[0].set_yticks(np.arange(0.6, 1.05, step=0.2))
  axs[0].set_xticks(np.arange(0, 64000, step=10000))
  axs[0].legend(loc='lower right', ncol=5)

  for i in range(5):
    axs[1].plot(data_points, known_acc_by_class[i], '-o', label='class {}'.format(i))
  axs[1].set_ylim([0.5, 1])
  axs[1].set_ylabel('Base classes', fontsize=12, rotation=0, ha='right')
  axs[1].set_yticks(np.arange(0.6, 1.05, step=0.2))
  axs[1].set_xticks(np.arange(0, 64000, step=10000))
  axs[1].legend(loc='lower right', ncol=5)

  for class_idx, class_acc in enumerate(known_acc_by_class[5:]):    
    axs[class_idx+2].plot(data_points, class_acc, '-o', color=colors[class_idx])
    axs[class_idx+2].axvline(x=start_points[class_idx], linestyle='--', color=colors[class_idx]) #color='k'
    axs[class_idx+2].axvline(x=detected_points[class_idx], linestyle='-.', color=colors[class_idx])
    axs[class_idx+2].axvspan(start_points[class_idx], detected_points[class_idx], alpha=0.25, color=colors[class_idx])

    axs[class_idx+2].set_ylim([0.5, 1])
    axs[class_idx+2].set_ylabel('Label {}'.format(class_idx+5), fontsize=12, rotation=0, ha='right')
    axs[class_idx+2].set_yticks(np.arange(0.6, 1.05, step=0.2)) 
    axs[class_idx+2].set_xticks(np.arange(0, 64000, step=5000))

    axs[class_idx+2].annotate(
      "",
      xy=(start_points[class_idx], 0.75), xycoords='data',
      xytext=(detected_points[class_idx], 0.75), textcoords='data',
      arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color=colors[class_idx], lw=1),
    )
    axs[class_idx+2].text(
      int(0.5*(detected_points[class_idx] + start_points[class_idx]) - 730 ), 0.8,
      '%g'%(detected_points[class_idx] - start_points[class_idx]), 
      rotation=0, fontsize=10, color=colors[class_idx])

    if class_idx != 4:
      axs[class_idx+2].set_xticklabels(())
  
    plt.xlabel('Stream data')
  
  plt.savefig('trajectory_eval.png', dpi=800)
  plt.show()


if __name__ == '__main__':
  # UDA_plot()
  avg_class_plot()
