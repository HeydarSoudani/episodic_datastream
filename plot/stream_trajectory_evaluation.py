import numpy as np
import matplotlib.pyplot as plt

def get_data():
  # mn_tr, mn_pt, fm_co, fm_pt, c10_tr, c10_rp
  data = {
    "mn_pt": {
      "x_points": np.array([5107, 9999, 12061, 19999, 21452, 29999, 33290, 39999, 49561, 49999, 59999, 63999]),
      'start_points': np.array([3001, 8001, 10012, 16000, 26014]),
      'detected_points': np.array([5107, 12061, 12061, 21452, 33290]),
      'CwCA': np.array([0.8406, 0.8831, 0.8731, 0.8766, 0.8602, 0.8812, 0.8628, 0.9012, 0.8697, 0.8950, 0.8860, 0.8838]),
      'Mnew': np.array([0.3159, 0.4118, 0.3744, 0.6840, 0.4759, 0.8487, 0.6875, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'Fnew': np.array([0.1519, 0.0738, 0.1145, 0.0801, 0.1169, 0.0649, 0.1030, 0.0356, 0.0796, 0.0434, 0.0681, 0.0650]),
      'acc_per_class': np.array([
        [0.8241, 0.9059, 0.8635, 0.9378, 0.8846, 0.9204, 0.904 , 0.962 , 0.9119,    None,    None,    None],
        [0.8888, 0.93  , 0.9381, 0.9466, 0.9419, 0.9618, 0.94  , 0.976 , 0.9635, 0.9718, 0.9359,    None],
        [0.814 , 0.8746, 0.8483, 0.8889, 0.8953, 0.8655, 0.824 , 0.9199, 0.8696,    None,    None,    None],
        [0.8482, 0.8847, 0.9034, 0.8698, 0.9048, 0.8661, 0.8587, 0.8814, 0.8333,    None,    None,    None],
        [0.8364, 0.9146, 0.8644, 0.9172, 0.887 , 0.9272, 0.9189, 0.8955, 0.8715, 0.8529,    None,    None],
        [   None, 0.7376, 0.7771, 0.8101, 0.779 , 0.8302, 0.8157, 0.8804, 0.8576, 0.8378, 0.8801,    None],
        [   None,    None,    None, 0.8202, 0.7989, 0.8374, 0.854 , 0.9224, 0.8981, 0.9481, 0.9148,    None],
        [   None,    None,    None, 0.83  , 0.784 , 0.9138, 0.8801, 0.9031, 0.8728,    0.8728, 0.9023, 0.8802],
        [   None,    None,    None,    None,    None, 0.7886, 0.7983, 0.8663, 0.8343, 0.9014, 0.8415, 0.8536],
        [   None,    None,    None,    None,    None,    None,    None, 0.7925, 0.8093, 0.8559, 0.8704, 0.9061]]
      )
    },
    "mn_tr": {
      "x_points": np.array([4682, 9999, 11336, 16030, 19999, 20566, 27009, 29999, 33204, 39562, 39999, 45693, 49999, 53703, 59999, 63448, 63999]),
      'start_points': np.array([3001, 8001, 10012, 16000, 26014]),
      'detected_points': np.array([4682, 11336, 16030, 20566, 33204]),
      'CwCA': np.array([0.8180, 0.8336, 0.8323, 0.8398, 0.8241, 0.7798, 0.8117, 0.8401, 0.8223, 0.8031, 0.8558, 0.8031, 0.8609, 0.8386, 0.8696, 0.8745, 0.9655]),
      'Mnew': np.array([0.3333, 0.6471, 0.3618, 0.3290, 0.3313, 0.2500, 0.5446, 0.6222, 0.6760, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'Fnew': np.array([0.1785, 0.1123, 0.1503, 0.1337, 0.1491, 0.1960, 0.1499, 0.1277, 0.1567, 0.1573, 0.1190, 0.1665, 0.1115, 0.1404, 0.1015, 0.1047, 0.0145]),
      'acc_per_class': np.array([
        [0.8064, 0.8357, 0.8032, 0.8517, 0.8306,    None, 0.8108, 0.8632, 0.8512, 0.8452, 0.8864, 0.8337, 0.8438,    None,    None,    None, None],
        [0.8733, 0.9177, 0.9351, 0.9418, 0.957 , 0.8788, 0.9209, 0.9337, 0.905 , 0.9463, 0.9388, 0.935 , 0.9395, 0.905 , 0.9126,    None, None],
        [0.7873, 0.8219, 0.8316, 0.7826, 0.796 , 0.7625, 0.781 , 0.8333, 0.7778, 0.7437, 0.8043, 0.7663, 0.7702,    None,    None,    None, None],
        [0.819 , 0.8127, 0.8503, 0.8318, 0.7513, 0.7231, 0.7736, 0.7771, 0.7961, 0.8039, 0.8605, 0.7592, 0.7582,    None,    None,    None, None],
        [0.815 , 0.8648, 0.8191, 0.8414, 0.8672, 0.8   , 0.8422, 0.8769, 0.8348, 0.8018,    None, 0.777 , 0.7819,    None,    None,    None, None],
        [   None, 0.7238, 0.7324, 0.8   , 0.7931, 0.7922, 0.7567, 0.753 , 0.746 , 0.7222, 0.8163, 0.7124, 0.8183, 0.7925, 0.8366,    None, None],
        [   None,    None,    None, 0.8229, 0.8495, 0.8308, 0.8606, 0.8799, 0.8563, 0.8619, 0.8667, 0.8378, 0.8748, 0.8505, 0.8454,    None, None],
        [   None,    None,    None,    None, 0.752 , 0.6716, 0.761 , 0.8198, 0.8479, 0.7067, 0.7321, 0.7516, 0.7979, 0.7948, 0.7935, 0.7975, None],
        [   None,    None,    None,    None,    None,    None, 0.7509, 0.8152, 0.7994, 0.8589, 0.875 , 0.8369, 0.9238, 0.8358, 0.9073, 0.8652, 0.8913],
        [   None,    None,    None,    None,    None,    None,    None,    None, None, 0.7143, 0.94  , 0.8312, 0.8736, 0.8488, 0.9323, 0.9332, 0.9903]])
    },
    "fm_pt": {
      "x_points": np.array([9999, 10011, 19999, 20876, 26887, 29999, 38316, 39999, 49999, 53879, 59999, 63999]),
      'start_points': np.array([12000, 16003, 19087, 21008, 23001]),
      'detected_points': np.array([20876, 29999, 29999, 26887, 38316]),
      'CwCA': np.array([0.8437, 0.7500, 0.8901, 0.8994, 0.8880, 0.8792, 0.8024, 0.8039, 0.7701, 0.7853, 0.8000, 0.8055]),
      'Mnew': np.array([0.0000, 0.0000, 0.6949, 0.4395, 0.6394, 0.9015, 0.6562, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'Fnew': np.array([0.0997, 0.2500, 0.0298, 0.0443, 0.0550, 0.0610, 0.0614, 0.0362, 0.0683, 0.0660, 0.0355, 0.0400]),
      'acc_per_class': np.array([
        [0.865 , 1.    , 0.9048, 0.928 , 0.9382, 0.9397, 0.8641, 0.8883, 0.8485,    None,    None,    None],
        [0.911 ,    None, 0.9591, 0.958 , 0.9561, 0.9614, 0.9617, 0.96  , 0.9775,    None,    None,    None],
        [0.7995, 1.    , 0.8553, 0.8492, 0.859 , 0.8798, 0.8197, 0.8385, 0.8545,    None,    None,    None],
        [0.8335, 0.3333, 0.877 , 0.8661, 0.8825, 0.8646, 0.8467, 0.9375, None,    None,    None,    None],
        [0.8095, 0.8   , 0.8586,    None, 0.8484, 0.7971, 0.8247, 0.8077, 0.7552,    None,    None,    None],
        [   None,    None,    None,    None, 0.8278, 0.922 , 0.8397, 0.8316, 0.8561, 0.8973, 0.932 ,    None],
        [   None,    None,    None,    None,    None,    None, 0.235 , 0.2356, 0.3275, 0.3987, 0.3527, 0.4225],
        [   None,    None,    None,    None,    None,    None, 0.8911, 0.9122, 0.8753, 0.8618, 0.8923, 0.895 ],
        [   None,    None,    None,    None,    None,    0.8174, 0.7587, 0.9167, 0.9117, 0.9176, 0.9444, 0.934 ],
        [   None,    None,    None,    None,    None,    None,    None, 0.8893, 0.8249, 0.8503, 0.8848, 0.9203]
      ])
    },
    "fm_co": {
      "x_points": np.array([8512, 9999, 17761, 19999, 25045, 29999, 35907, 39999, 47896, 49999, 59999, 62003, 63999]),
      'start_points': np.array([12000, 16003, 19087, 21008, 23001]),
      'detected_points': np.array([17761, 29999, 25045, 25045, 29999]),
      'CwCA': np.array([0.8265, 0.8890, 0.8782, 0.8501, 0.8528, 0.8446, 0.7877, 0.8343, 0.7807, 0.8398, 0.8505, 0.7999, 0.8823]),
      'Mnew': np.array([0.0000, 0.0000, 0.4574, 0.9552, 0.6674, 0.7299, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'Fnew': np.array([0.1175, 0.0457, 0.0586, 0.0787, 0.0796, 0.0953, 0.0633, 0.0494, 0.1011, 0.0509, 0.0633, 0.1297, 0.0386]),
      'acc_per_class': np.array([
        [0.8148, 0.8741, 0.8794, 0.8817, 0.8641, 0.8802, 0.8043, 0.8447, 0.7879,    None,    None,    None,    None],
        [0.8703, 0.9585, 0.9424, 0.9322, 0.9448, 0.9253, 0.9315, 0.9522, 0.9663,    None,    None,    None,    None],
        [0.8491, 0.8384, 0.866 , 0.8626, 0.8381, 0.789 , 0.7378, 0.7813, 0.6909,    None,    None,    None,    None],
        [0.8101, 0.898 , 0.8538, 0.8918, 0.8837, 0.8809, 0.8258, 0.875 , None,    None,    None,    None,    None],
        [0.7879, 0.8779, 0.8565, 0.8344, 0.8643, 0.8694, 0.7081, 0.6967, 0.7229,    None,    None,    None,    None],
        [   None,    None,    None, 0.707 , 0.7339, 0.8907, 0.8435, 0.9043, 0.8228, 0.875 , 0.8759,    None,    None],
        [   None,    None,    None,    None,    None,    None, 0.4927, 0.4653, 0.5507, 0.6312, 0.7065, 0.684 , 0.6902],
        [   None,    None,    None,    None,    None,    0.7699, 0.8275, 0.9038, 0.8304, 0.8659, 0.854 , 0.784 , 0.908 ],
        [   None,    None,    None,    None,    None,    0.7559, 0.7691, 0.8404, 0.8247, 0.9165, 0.906 , 0.8962, 0.9291],
        [   None,    None,    None,    None,    None,    None, 0.8733, 0.8987, 0.8544, 0.911 , 0.9078, 0.835 , 0.9467]
      ])
    },
    "c10_rp": {
      "x_points": np.array([4999, 9999, 11409, 14999, 19999, 24999, 26103, 29999, 34999, 39999, 44999, 49999, 50337, 53999]),
      'start_points': np.array([7001, 12021, 14028, 17009, 18009]),
      'detected_points': np.array([14999, 26103, 26103, 26103, 26103]),
      'CwCA': np.array([0.6132, 0.5934, 0.6236, 0.6636, 0.5625, 0.5138, 0.5284, 0.3832, 0.4476, 0.4872, 0.5018, 0.5318, 0.5355, 0.5945]),
      'Mnew': np.array([0.0000, 0.8850, 0.8506, 0.9795, 0.9497, 0.8912, 0.8604, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'Fnew': np.array([0.0638, 0.1077, 0.0945, 0.0378, 0.0581, 0.0844, 0.1070, 0.0149, 0.0268, 0.0372, 0.0484, 0.0706, 0.0799, 0.0180]),
      'acc_per_class': np.array([
        [0.708 , 0.6775, 0.695 , 0.737 , 0.6735, 0.6252, 0.6   , 0.4768, 0.3996, 0.4854,    None,    None,    None,    None],
        [0.733 , 0.7275, 0.7754, 0.7709, 0.7648, 0.7387, 0.627 , 0.4937, 0.542 , 0.5422,    None,    None,    None,    None],
        [0.332 , 0.362 , 0.4484, 0.452 , 0.4152, 0.445 , 0.4715, 0.4097, 0.433 ,    None,    None,    None,    None,    None],
        [0.654 , 0.603 , 0.5338, 0.5497, 0.4229, 0.4054, 0.3852, 0.2171, 0.2645,    None,    None,    None,    None,    None],
        [0.639 , 0.6525, 0.6877, 0.7844, 0.7575, 0.7207, 0.7568, 0.6306, 0.6165, 0.619 ,    None,    None,    None,    None],
        [   None,    None,    None,    None, 0.3154, 0.2757, 0.416 , 0.4279, 0.4134, 0.4545, 0.417 , 0.429 , 0.3065, 0.4196],
        [   None,    None,    None,    None,    None,    None,    None, 0.2564, 0.4258, 0.5074, 0.571 , 0.601 , 0.6349, 0.6466],
        [   None,    None,    None,    None,    None,    None,    None, 0.3819, 0.4651, 0.4718, 0.495 , 0.533 , 0.64  , 0.6182],
        [   None,    None,    None,    None,    None,    None,    None, 0.2959, 0.3756, 0.4105, 0.512 , 0.536 , 0.459 , 0.559 ],
        [   None,    None,    None,    None,    None,    None,    None, 0.4165, 0.4658, 0.4983, 0.514 , 0.56  , 0.5974, 0.5867]
      ])
    },
    "c10_tr": {
      "x_points": np.array([4999, 7666, 9999, 14999, 17226, 19999, 22428, 24999, 29999, 34999, 37794, 39999, 44999, 49726, 49999, 53999]),
      'start_points': np.array([7001, 12021, 14028, 17009, 18009]),
      'detected_points': np.array([17226, 22428, 22428, 22428, 22428]),
      'CwCA': np.array([0.6758, 0.6357, 0.6950, 0.6944, 0.6465, 0.5519, 0.5317, 0.4403, 0.4940, 0.5326, 0.5635, 0.6372, 0.6554, 0.6444, 0.7326, 0.7270]),
      'Mnew': np.array([0.0000, 0.8676, 0.9267, 0.9380, 0.8508, 0.7884, 0.7232, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
      'Fnew': np.array([0.1170, 0.1569, 0.0770, 0.1084, 0.1505, 0.1578, 0.1614, 0.0292, 0.0540, 0.0688, 0.1113, 0.0558, 0.0596, 0.1225, 0.0659, 0.0847]),
      'acc_per_class': np.array([
        [0.708 , 0.6748, 0.7208, 0.7087, 0.6367, 0.5   , 0.5222, 0.4386, 0.4895, 0.441 , 0.4929, 0.4615,    None,    None,    None,    None],
        [0.749 , 0.6725, 0.735 , 0.722 , 0.7186, 0.6366, 0.674 , 0.4795, 0.5135, 0.5214, 0.5301,    None,    None,    None,    None,    None],
        [0.558 , 0.5717, 0.6562, 0.6384, 0.6505, 0.4689, 0.5152, 0.457 , 0.4937, 0.5128,    None,    None,    None,    None,    None,    None],
        [0.688 , 0.666 , 0.6659, 0.6613, 0.5915, 0.3468, 0.3664, 0.3481, 0.3495, 0.371 ,    None,    None,    None,    None,    None,    None],
        [0.676 , 0.6015, 0.73  , 0.7368, 0.6635, 0.655 , 0.5856, 0.4414, 0.5135, 0.4788, 0.5481, 0.4023,    None,    None,    None,    None],
        [   None,    None,    None,    None,    None, 0.6942, 0.5519, 0.3298, 0.3766, 0.4443, 0.4577, 0.506 , 0.51  , 0.535 , 0.7321, 0.6634],
        [   None,    None,    None,    None,    None,    None,    None, 0.4373, 0.6   , 0.697 , 0.6507, 0.691 , 0.72  , 0.6854, 0.825 , 0.7321],
        [   None,    None,    None,    None,    None,    None,    None, 0.431 , 0.4919, 0.5306, 0.5697, 0.61  , 0.633 , 0.5937, 0.6364, 0.6633],
        [   None,    None,    None,    None,    None,    None,    None, 0.5379, 0.6144, 0.6827, 0.6034, 0.7325, 0.769 , 0.7489, 0.7667, 0.7989],
        [   None,    None,    None,    None,    None,    None,    None, 0.517 , 0.5071, 0.5575, 0.5617, 0.6516, 0.645 , 0.6588, 0.7258, 0.728 ]
      ])
    }
  }
  
  return data

def UDA_plot():
  method = 'mn_tr' #[mn_tr, mn_pt, fm_co, fm_pt, c10_tr, c10_rp]
  data = get_data()

  fig, axs = plt.subplots(3, 2)

  for idx, (method, item) in enumerate(data.items()):
    
    if method in ['mn_tr', 'mn_pt', 'fm_pt', 'fm_co']:
      x_lim = [0, 65000]
      x_ticks = np.arange(0, 64000, step=10000)
    elif method in ['c10_rp', 'c10_tr']:
      x_lim = [0, 55000]
      x_ticks = np.arange(0, 54000, step=5000)
    
    # print(item)
    x_points = item['x_points']
    Mnew = item['Mnew']
    Fnew = item['Fnew']

    axs[int(idx/2), idx%2].plot(x_points, Mnew, '-o', label='M_new')
    axs[int(idx/2), idx%2].plot(x_points, Fnew, '-o', label='F_new')

    axs[int(idx/2), idx%2].set_title('Method: {}'.format(method))
    axs[int(idx/2), idx%2].set_ylim([0.0, 1])
    axs[int(idx/2), idx%2].set_yticks(np.arange(0.0, 1.05, step=0.25))
    axs[int(idx/2), idx%2].set_ylabel('Percentage', fontsize=10)
    axs[int(idx/2), idx%2].set_xlim(x_lim)
    axs[int(idx/2), idx%2].set_xticks(x_ticks)
    axs[int(idx/2), idx%2].set_xlabel('Stream data')

  handles, labels = axs[0,0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', ncol=2)
  fig.subplots_adjust(hspace=0.4)
  plt.show()

def avg_class_plot():
  dataset = 'CIFAR10' #['MNIST', 'FashionMNIST', 'CIFAR10',]
  class_num = 10
  method = 'c10_tr' #[mn_tr, mn_pt, fm_co, fm_pt, c10_tr, c10_rp]
  data = get_data()[method]
  colors = ['limegreen', 'hotpink', 'blueviolet', 'royalblue', 'darkorange', 'gold', 'brown']

  ### Get data
  x_points = data['x_points']
  start_points = data['start_points']
  detected_points = data['detected_points']
  acc_per_class = data['acc_per_class']
  CwCA = data['CwCA']
  
  if method in ['mn_tr', 'mn_pt']:
    y_lim = [0.5, 1]
    x_lim = [0, 65000]
    y_ticks = np.arange(0.6, 1.05, step=0.2)
    x_ticks = np.arange(0, 64000, step=10000)
    y_annotate = 0.75
  elif method in ['fm_pt', 'fm_co']:
    y_lim = [0.2, 1]
    x_lim = [0, 65000]
    y_ticks = np.arange(0.3, 1.05, step=0.2)
    x_ticks = np.arange(0, 64000, step=10000)
    y_annotate = 0.6
  elif method in ['c10_rp', 'c10_tr']:
    y_lim = [0.2, 0.85]
    x_lim = [0, 55000]
    y_ticks = np.arange(0.3, 0.8, step=0.2)
    x_ticks = np.arange(0, 54000, step=5000)
    y_annotate = 0.5

  # plt.rcParams['axes.grid'] = True
  fig, axs = plt.subplots(7, 1)
  fig.subplots_adjust(hspace = .001)
  plt.suptitle('{} dataset, {}'.format(dataset, method), fontsize=14, y=0.92)

  axs[0].plot(x_points, CwCA, '-o', label='CwCA', color=colors[-1])
  axs[0].set_ylim(y_lim)
  axs[0].set_xlim(x_lim)
  axs[0].set_ylabel('Known class Acc.', fontsize=12, rotation=0, ha='right')
  axs[0].set_yticks(y_ticks)
  axs[0].set_xticks(x_ticks)
  axs[0].legend(loc='lower right', ncol=5)

  for i in range(5):
    axs[1].plot(x_points, acc_per_class[i], '-o', label='class {}'.format(i))
  axs[1].set_ylim(y_lim)
  axs[1].set_xlim(x_lim)
  axs[1].set_ylabel('Base classes', fontsize=12, rotation=0, ha='right')
  axs[1].set_yticks(y_ticks)
  axs[1].set_xticks(x_ticks)
  axs[1].legend(loc='lower right', ncol=5)

  for class_idx, class_acc in enumerate(acc_per_class[5:]):    
    axs[class_idx+2].plot(x_points, class_acc, '-o', color=colors[class_idx])
    axs[class_idx+2].axvline(x=start_points[class_idx], linestyle='--', color=colors[class_idx]) #color='k'
    axs[class_idx+2].axvline(x=detected_points[class_idx], linestyle='-.', color=colors[class_idx])
    axs[class_idx+2].axvspan(start_points[class_idx], detected_points[class_idx], alpha=0.25, color=colors[class_idx])

    axs[class_idx+2].set_ylim(y_lim)
    axs[class_idx+2].set_xlim(x_lim)
    axs[class_idx+2].set_ylabel('Label {}'.format(class_idx+5), fontsize=12, rotation=0, ha='right')
    axs[class_idx+2].set_yticks(y_ticks) 
    axs[class_idx+2].set_xticks(x_ticks)

    axs[class_idx+2].annotate(
      "",
      xy=(start_points[class_idx], y_annotate), xycoords='data',
      xytext=(detected_points[class_idx], y_annotate), textcoords='data',
      arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color=colors[class_idx], lw=1),
    )
    axs[class_idx+2].text(
      int(0.5*(detected_points[class_idx] + start_points[class_idx]) - 730 ), y_annotate+0.05,
      '%g'%(detected_points[class_idx] - start_points[class_idx]), 
      rotation=0, fontsize=10, color=colors[class_idx])

    if class_idx != 4:
      axs[class_idx+2].set_xticklabels(())
  
    plt.xlabel('Stream data')
  
  plt.savefig('trajectory_eval.png', dpi=800)
  plt.show()


if __name__ == '__main__':
  UDA_plot()
  # avg_class_plot()
