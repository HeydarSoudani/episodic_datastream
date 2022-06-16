import torch
# import torchvision
# import torchvision.transforms as transforms
# from scipy import ndimage
# import matplotlib.pyplot as plt
# import pandas as pd 
import numpy as np
# import argparse
# import os

# =
97.54,    5.00,   13.66
100.00,   9.06,   15.98
99.72,    6.22,   13.00

99.09,    6.76,   14.21

# =
98.80,    3.47,   14.27
96.01,    7.04,   12.65
97.91,   22.15,   14.12

97.57,   10.89,  13.68

# =
97.78,   37.56,   14.81
98.21,    9.32,   14.77
96.83,   15.46,   14.66

97.61,   20.78,   14.75

# =
95.72,   20.78,   15.65
97.03,   23.25,   17.58
95.17,   37.57,   17.38

95.97,   27.20,   16.87

# =
93.84,    0.00,   18.52
94.61,    0.00,   17.56
92.62,    0.00,   19.15

93.69,         0, 18.41


##### FashionMNIST

# =
99.15,   29.15,   15.40
98.05,   44.20,   12.90
96.95,   23.90,   13.95
99.30,   59.05,   15.00

98.36, 39.08, 14.31

# =
90.97,   82.70,   16.42
90.85,   71.15,   11.40
92.50,   58.55,   14.92
90.35,   60.25,   14.70

91.17,  68.16,  14.36

# =
83.17,   85.15,   15.57
84.02,   74.85,   12.05
86.68,   89.25,   14.43
87.80,   86.80,   16.77

85.42, 84.01,    14.70 

# =
73.78,   87.40,   12.01
60.36,   78.05,   18.66
70.06,   74.15,   13.75
70.70,   81.30,   21.85

68.72,   80.22,   16.57

a = [13.36, 26.70, 16.35, 13.31]
print('{:.2f} ± {:.2f}'.format(np.mean(a), np.std(a)))

# =
67.79,    0.00,   13.36
66.97,    0.00,   26.70
78.33,    0.00,   16.35
78.93,    0.00,   13.31

73.00, 0 ,    17.43 

# a = [
#   25.74,
#   24.57,
#   17.88,
#   # 0.6191
# ]
# print('{:.2f} ± {:.2f}'.format(np.mean(a), np.std(a)))
# print('{:.2f} ± {:.2f}'.format(np.mean(a)*100, np.std(a)*100))

## -- Forgetting ----------------
# all_tasks_acc_dist = torch.tensor([
#   [0.9775, 0.0, 0.0, 0.0, 0.0],
#   [0.9595, 0.888, 0.0, 0.0, 0.0],
#   [0.923, 0.807, 0.8435, 0.0, 0.0],
#   [0.7945, 0.667, 0.811, 0.872, 0.0],
#   [0.8585, 0.7625, 0.8095, 0.717, 0.9805]
# ])
# n_tasks = 5
# # all_tasks_acc_dist = np.transpose(b)
# acc_dist_best = torch.max(all_tasks_acc_dist, 0).values
# temp = acc_dist_best - all_tasks_acc_dist
# # print(temp)
# forgetting_dist = torch.tensor([torch.mean(temp[i+1:, i]) for i in range(n_tasks-1)])
# mean_forgetting_dist = torch.mean(forgetting_dist)
# std_forgetting_dist = torch.std(forgetting_dist)
# print('dist forgetting: {:.2f} ± {:.2f}'.format(mean_forgetting_dist*100, std_forgetting_dist*100))



# a = np.transpose(b)
# print(a)
# f = 0.0
# for item in a:
#   f += a[0] - item
# f /= (len(a)-1)
# print(f)


# n_task = 5
# b = [
#   {
#     'task_0': [99.27, 99.18, 99.64, 99.73, 99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.64, 99.68, 99.64, 99.82, 99.82, 99.82, 99.91, 99.82, 99.91, 99.91, 99.82, 99.91, 99.91, 99.91, 99.82, 99.91, 99.91, 99.82, 87.01, 93.6, 96.02, 96.88, 96.66, 96.8, 97.71, 97.3, 97.66, 97.48, 96.74, 97.25, 97.3, 97.29, 97.11, 97.29, 96.61, 96.89, 96.61, 97.02, 97.2, 96.57, 96.42, 97.25, 96.98, 96.93, 96.79, 96.93, 96.19, 96.01, 81.97, 87.11, 87.57, 89.96, 91.51, 91.47, 92.11, 92.66, 93.3, 92.71, 93.67, 93.82, 94.09, 93.53, 94.73, 93.58, 93.4, 93.86, 92.52, 94.0, 93.34, 93.67, 93.85, 94.36, 94.31, 93.94, 93.85, 92.81, 94.41, 94.68, 87.71, 85.27, 83.02, 84.67, 83.71, 85.59, 86.69, 87.98, 86.51, 86.92, 86.1, 85.49, 87.15, 87.42, 87.98, 88.39, 89.03, 88.94, 90.83, 89.95, 89.26, 91.56, 91.93, 91.01, 89.63, 90.92, 91.06, 92.11, 91.1, 92.99, 85.49, 81.79, 82.29, 82.47, 82.48, 85.32, 86.47, 86.48, 86.52, 86.7, 87.21, 89.09, 88.45, 88.4, 88.9, 87.76, 88.17, 88.86, 89.82, 88.95, 88.49, 89.55, 89.5, 89.05, 89.27, 89.4, 88.9, 89.22, 88.36, 87.53],
#     'task_1': [84.86, 95.58, 97.1, 97.44, 98.28, 98.57, 98.62, 98.97, 99.01, 99.01, 99.16, 99.31, 99.4, 99.31, 99.31, 99.26, 99.41, 99.31, 99.56, 99.56, 99.61, 99.6, 99.46, 99.51, 99.41, 99.6, 99.65, 99.56, 99.66, 99.46, 69.38, 76.93, 78.16, 80.8, 80.69, 84.26, 83.91, 86.51, 87.19, 88.46, 88.16, 89.14, 89.15, 89.45, 89.16, 89.83, 88.61, 90.87, 90.03, 90.38, 91.16, 89.45, 89.99, 90.28, 90.32, 91.0, 91.46, 91.54, 92.09, 90.54, 75.73, 78.13, 77.59, 78.46, 79.04, 79.49, 80.52, 80.76, 82.19, 80.29, 80.92, 81.51, 79.94, 81.17, 82.16, 81.46, 82.78, 82.98, 82.59, 81.95, 83.38, 83.57, 83.48, 82.55, 82.5, 82.89, 83.82, 82.99, 83.62, 84.7, 71.96, 73.57, 72.32, 75.21, 77.17, 78.09, 79.16, 79.51, 80.14, 80.25, 80.54, 79.6, 79.9, 80.04, 80.24, 80.64, 81.17, 81.41, 81.36, 81.51, 80.63, 82.24, 81.46, 81.66, 82.24, 80.72, 81.27, 80.83, 80.73, 82.1],
#     'task_2': [85.88, 90.62, 93.01, 95.58, 95.99, 96.89, 96.91, 97.3, 97.22, 98.09, 98.72, 98.3, 98.8, 98.45, 98.19, 98.4, 98.35, 98.24, 98.48, 98.45, 98.59, 98.74, 98.63, 98.06, 98.32, 98.45, 99.0, 98.79, 98.77, 98.35, 70.61, 77.36, 77.7, 81.42, 81.83, 81.49, 80.53, 81.57, 80.92, 80.76, 81.08, 81.34, 82.85, 82.51, 84.37, 82.48, 84.69, 87.11, 86.14, 86.63, 86.38, 88.13, 87.07, 86.61, 85.39, 86.58, 87.24, 86.82, 88.68, 89.7, 71.98, 73.04, 71.03, 72.91, 75.21, 74.95, 77.13, 77.65, 77.79, 76.84, 76.79, 77.36, 78.39, 78.16, 78.11, 78.82, 79.23, 77.62, 77.87, 77.01, 76.63, 77.7, 78.35, 78.63, 77.16, 76.78, 77.52, 78.56, 77.26, 77.25],
#     'task_3': [92.53, 94.98, 96.05, 95.12, 94.78, 95.02, 94.63, 94.48, 95.12, 94.97, 95.26, 95.31, 95.75, 95.9, 95.9, 95.8, 96.0, 95.56, 95.9, 95.9, 96.04, 95.95, 96.14, 95.95, 95.95, 95.65, 95.85, 95.7, 95.21, 96.09, 77.69, 76.28, 79.59, 80.65, 82.64, 84.39, 86.05, 86.0, 86.97, 87.41, 87.8, 87.22, 86.98, 87.8, 88.04, 88.53, 88.49, 88.82, 88.49, 88.78, 89.12, 89.07, 88.83, 88.39, 87.91, 88.05, 87.62, 87.37, 87.86, 87.76],
#     'task_4': [92.28, 95.21, 95.6, 96.33, 96.87, 97.31, 97.21, 97.65, 97.7, 97.85, 98.05, 97.36, 98.24, 98.54, 98.58, 98.63, 98.54, 98.49, 98.54, 98.63, 98.49, 98.93, 98.73, 98.93, 98.97, 98.68, 98.78, 98.97, 99.02, 99.17]
#   },
#   {'task_0': [99.36, 99.54, 99.73, 99.77, 99.82, 99.68, 99.73, 99.82, 99.68, 99.77, 99.73, 99.91, 99.91, 99.77, 99.91, 99.77, 99.77, 99.82, 99.73, 99.82, 99.82, 99.82, 99.73, 99.82, 99.91, 99.73, 99.77, 99.77, 99.91, 99.91, 86.05, 93.76, 95.51, 96.24, 96.19, 96.93, 96.79, 96.74, 96.61, 97.02, 96.98, 97.03, 97.2, 97.11, 97.34, 97.16, 96.93, 97.07, 97.07, 96.79, 96.79, 96.79, 96.89, 96.88, 96.47, 95.7, 96.11, 96.01, 96.15, 96.29, 83.2, 86.75, 87.07, 87.89, 89.45, 90.55, 91.24, 91.66, 91.93, 92.1, 92.07, 92.89, 92.76, 92.56, 92.25, 92.62, 92.99, 93.13, 92.98, 92.62, 92.29, 92.29, 93.44, 93.34, 92.93, 93.16, 93.44, 93.16, 93.44, 93.66, 83.33, 82.46, 82.7, 84.49, 84.26, 84.86, 85.1, 84.78, 86.14, 87.79, 88.08, 88.53, 90.45, 90.45, 90.87, 90.46, 91.73, 90.82, 91.37, 91.82, 91.87, 91.42, 92.34, 92.57, 91.38, 91.79, 92.11, 91.74, 91.47, 92.53, 83.91, 84.29, 82.77, 84.08, 82.46, 82.51, 85.62, 87.13, 85.72, 86.27, 86.4, 87.73, 87.74, 87.84, 88.12, 88.52, 87.1, 88.29, 88.92, 89.07, 88.69, 86.42, 88.28, 89.42, 89.0, 89.47, 89.97, 89.78, 89.51, 90.06], 'task_1': [93.32, 96.41, 97.4, 97.53, 97.69, 97.78, 98.13, 98.47, 98.82, 98.62, 98.82, 99.22, 99.02, 99.36, 99.07, 98.97, 99.02, 98.92, 99.41, 99.16, 98.97, 99.02, 99.16, 99.31, 99.51, 99.46, 99.36, 99.31, 99.41, 99.41, 62.03, 74.66, 79.42, 83.72, 84.9, 87.5, 89.45, 88.81, 89.94, 90.09, 91.91, 91.76, 92.44, 91.41, 92.05, 90.73, 91.47, 92.88, 91.75, 92.0, 91.76, 91.17, 91.56, 91.12, 92.59, 92.49, 92.69, 92.2, 92.19, 92.54, 77.2, 78.18, 77.25, 80.97, 80.97, 81.75, 80.3, 82.2, 81.32, 81.17, 83.32, 82.89, 84.3, 83.47, 83.52, 82.45, 84.46, 85.73, 84.7, 85.39, 84.99, 85.98, 86.31, 86.52, 85.69, 85.73, 85.73, 87.0, 86.07, 86.56, 73.57, 72.44, 70.05, 74.01, 74.6, 75.19, 76.12, 77.34, 76.85, 78.96, 78.62, 77.54, 79.35, 79.88, 79.12, 80.39, 78.57, 78.13, 80.58, 79.6, 80.88, 80.53, 80.14, 80.09, 79.5, 79.89, 79.94, 80.29, 79.7, 79.7], 'task_2': [86.92, 93.43, 93.46, 96.66, 94.36, 96.62, 97.64, 98.04, 97.93, 97.01, 97.04, 98.04, 97.93, 97.12, 98.51, 98.07, 96.94, 98.33, 97.54, 98.51, 98.12, 97.55, 98.17, 96.91, 98.77, 97.84, 98.77, 96.16, 98.17, 96.94, 77.01, 82.12, 81.88, 84.83, 82.41, 82.51, 84.16, 83.38, 83.08, 86.52, 87.64, 89.01, 91.56, 91.98, 91.92, 90.85, 93.44, 92.58, 91.84, 92.76, 90.43, 91.73, 93.41, 92.71, 94.09, 91.32, 94.11, 93.25, 91.55, 93.32, 71.14, 71.19, 72.05, 69.52, 71.96, 72.56, 72.77, 73.79, 73.33, 75.58, 78.87, 76.16, 77.04, 78.71, 78.66, 79.27, 79.31, 79.55, 78.92, 79.5, 78.27, 79.05, 79.41, 79.2, 79.28, 78.43, 79.26, 80.3, 80.46, 80.15], 'task_3': [91.32, 93.41, 95.12, 95.31, 95.12, 95.26, 95.31, 95.31, 95.7, 95.61, 95.85, 96.14, 96.44, 96.24, 96.09, 96.48, 96.29, 96.48, 96.53, 96.92, 96.34, 96.78, 96.88, 97.07, 96.04, 96.68, 96.53, 96.58, 96.78, 96.78, 80.99, 80.76, 80.07, 80.8, 81.82, 81.96, 80.65, 82.06, 83.37, 83.76, 84.05, 84.44, 84.64, 85.32, 84.83, 84.84, 86.06, 86.88, 87.27, 85.23, 86.16, 86.25, 85.86, 85.28, 85.18, 85.37, 86.64, 85.71, 86.5, 85.81], 'task_4': [88.71, 90.03, 88.96, 91.35, 92.86, 93.84, 95.16, 95.11, 95.46, 95.8, 95.85, 96.63, 96.43, 96.63, 96.92, 96.78, 96.68, 96.88, 97.07, 96.83, 96.88, 96.73, 97.22, 97.02, 97.51, 97.07, 97.02, 96.97, 97.12, 97.12]}
# ]

# f = open('output1.txt', 'w')
# output = {
#   'mean': {},
#   'std': {}
# }

# for task in range(n_task):
#   temp = [item['task_{}'.format(task)]  for item in b]
#   output['mean']['task_{}'.format(task)] = np.around(np.mean(temp, axis=0), 2).tolist()
#   output['std']['task_{}'.format(task)] = np.around(np.std(temp, axis=0), 2).tolist()

# print(output)
# f.write(str(output))




### ---- read from .txt file for incremental -------
# from numpy import loadtxt
# # lines = loadtxt("output.txt", comments="#", delimiter=" ", unpack=False)

# text_file = open("output.txt", "r")
# lines = text_file.readlines()
# new_list = []
# for line in lines:
#   item = line[1:-2]
#   items = item.split(",")
#   float_items = [round(float(i)*100, 2) for i in items]
#   new_list.append(float_items)

# arr = np.array(new_list)
# n_task = 5
# n_epoch_item = 3

# output = {}
# for task in range(n_task):
#   output['task_{}'.format(task)] = list(arr[(task*n_epoch_item):, task])

# print(output)


### ---- for stream trajectory -------
# n_class = 10
# data = [
#   [0.715, 0.777, 0.524, 0.62,  0.672],
#   [0.7218, 0.70819672, 0.51548947, 0.60223048, 0.64344942],
#   [0.72875, 0.74210526, 0.59585492, 0.60103627, 0.70558376],
#   [0.71653543, 0.70366133, 0.62711864, 0.64530892, 0.69221968],
#   [0.60201511, 0.67322835, 0.50510204, 0.44366197, 0.6375, 0.46517413],
#   [0.66239316, 0.71489362, 0.58995816, 0.49099099, 0.71861472, 0.50847458],
#   [0.61261261, 0.73873874, 0.54594595, 0.41081081, 0.57057057, 0.45405405, 0.58108108],
#   [0.611, 0.63265306, 0.35555556, 0.37777778, 0.5, 0.43902439, 0.52,       0.54166667],
#   [0.60960961, 0.60506329, 0.49019608, 0.36666667, 0.58762887, 0.53501946, 0.57227723, 0.57593688, 0.48418972, 0.4844358],
#   [0.51746725, 0.6535163,  0.52991453, 0.33870968, 0.61016949, 0.51801029, 0.58898305, 0.51310044, 0.60891938, 0.49315068],
#   [0.59393939, 0.71084337,        None,        None, 0.52941176, 0.5,        0.62,       0.55405405, 0.62365591, 0.56300997],
#   [0.5,  None, None,  None,        None, 0.505,      0.6969697,  0.51315789, 0.72222222, 0.58252427],
#   [None,  None, None,  None,         None, 0.51,  0.672, 0.646, 0.709, 0.669],
#   [None,  None, None,  None,   None, 0.52,  0.685, 0.634, 0.71,  0.686],
#   [None,  None,  None, None,        None, 0.60696517, 0.65566038, 0.71634615, 0.68396226, 0.67699115,],
#   [None,  None,  None, None,        None, 0.75,       0.71323529, 0.70953757, 0.79505814, 0.72976055,]
# ]
# print(data)

# new_data = []
# for row in data:
#   new_row = np.concatenate(
#     (np.array(row),
#     np.full((n_class - len(row), ), 0.0))
#   )
#   new_data.append(new_row)

# new_data_arr = np.array(new_data)
# new_data_arr = np.round(new_data_arr.astype('float'), 4)
# new_data_arr[new_data_arr == 0.] = None
# new_data_arr_t = np.transpose(new_data_arr)

# print(repr(new_data_arr_t))

# a = np.array([
# [0.5, 0.0, 0.0, 0.0, 0.0],
# [0.9805, 0.0, 0.0, 0.0, 0.0],
# [0.9825, 0.0, 0.0, 0.0, 0.0],
# [0.969, 0.0, 0.0, 0.0, 0.0],
# [0.969, 0.0, 0.0, 0.0, 0.0],
# [0.988, 0.0, 0.0, 0.0, 0.0],
# [0.988, 0.0, 0.0, 0.0, 0.0],
# [0.991, 0.0, 0.0, 0.0, 0.0],
# [0.991, 0.0, 0.0, 0.0, 0.0],
# [0.9855, 0.0, 0.0, 0.0, 0.0],

# [0.8675, 0.8815, 0.0, 0.0, 0.0],
# [0.8985, 0.946, 0.0, 0.0, 0.0],
# [0.8985, 0.946, 0.0, 0.0, 0.0],
# [0.9545, 0.884, 0.0, 0.0, 0.0],
# [0.9545, 0.884, 0.0, 0.0, 0.0],
# [0.9535, 0.9365, 0.0, 0.0, 0.0],
# [0.9535, 0.9365, 0.0, 0.0, 0.0],
# [0.9605, 0.8595, 0.0, 0.0, 0.0],
# [0.9605, 0.8595, 0.0, 0.0, 0.0],
# [0.9765, 0.796, 0.0, 0.0, 0.0],

# [0.945, 0.7315, 0.901, 0.0, 0.0],
# [0.9565, 0.85, 0.8455, 0.0, 0.0],
# [0.9565, 0.85, 0.8455, 0.0, 0.0],
# [0.9295, 0.814, 0.9475, 0.0, 0.0],
# [0.9295, 0.814, 0.9475, 0.0, 0.0],
# [0.9175, 0.824, 0.9345, 0.0, 0.0],
# [0.9175, 0.824, 0.9345, 0.0, 0.0],
# [0.906, 0.8375, 0.9375, 0.0, 0.0],
# [0.906, 0.8375, 0.9375, 0.0, 0.0],
# [0.9075, 0.7755, 0.951, 0.0, 0.0],

# [0.9285, 0.7055, 0.751, 0.692, 0.0],
# [0.758, 0.809, 0.6915, 0.837, 0.0],
# [0.758, 0.809, 0.6915, 0.837, 0.0],
# [0.772, 0.8815, 0.721, 0.687, 0.0],
# [0.772, 0.8815, 0.721, 0.687, 0.0],
# [0.8405, 0.8545, 0.77, 0.736, 0.0],
# [0.8405, 0.8545, 0.77, 0.736, 0.0],
# [0.866, 0.7505, 0.859, 0.7645, 0.0],
# [0.866, 0.7505, 0.859, 0.7645, 0.0],
# [0.8545, 0.73, 0.861, 0.8005, 0.0],

# [0.915, 0.799, 0.8215, 0.65, 0.9485],
# [0.8515, 0.8565, 0.6755, 0.7165, 0.892],
# [0.8515, 0.8565, 0.6755, 0.7165, 0.892],
# [0.9045, 0.8655, 0.7075, 0.6225, 0.984],
# [0.9045, 0.8655, 0.7075, 0.6225, 0.984],
# [0.833, 0.7215, 0.925, 0.74, 0.8315],
# [0.833, 0.7215, 0.925, 0.74, 0.8315],
# [0.8775, 0.887, 0.6975, 0.749, 0.974],
# [0.8775, 0.887, 0.6975, 0.749, 0.974],
# [0.9145, 0.822, 0.8475, 0.6635, 0.979],
# ])
# b = np.transpose(a)
# n_task = 5
# output = {}
# for task in range(n_task):
#   output['task_{}'.format(task)] = (np.round(100*b[task, task*10:], 2)).tolist()

# print(output)


























# def imshow(imgs):

#   grid_imgs = torchvision.utils.make_grid(torch.tensor(imgs), nrow=2)
#   plt.imshow(grid_imgs.permute(1, 2, 0))
  
  
#   # # imgs = imgs / 2 + 0.5     # unnormalize
#   # npimg = imgs.detach().cpu().numpy()
#   # print(imgs.shape)
#   # # print(np.transpose(imgs, (1, 2, 0)).shape)
#   # plt.imshow(imgs)
#   # plt.imshow(np.transpose(npimg, (1, 2, 0)))
#   plt.show()


# ## == Params ==========================
# parser = argparse.ArgumentParser()
# parser.add_argument('--n_tasks', type=int, default=5, help='')
# parser.add_argument(
#   '--dataset',
#   type=str,
#   choices=[
#     'mnist',
#     'pmnist',
#     'rmnist',
#     'fmnist',
#     'cifar10'
#   ],
#   default='mnist',
#   help='')
# parser.add_argument('--seed', type=int, default=2, help='')
# args = parser.parse_args()

# # = Add some variables to args ========
# args.data_path = 'data/{}'.format('mnist' if args.dataset in ['pmnist', 'rmnist'] else args.dataset)
# args.train_path = 'train'
# args.test_path = 'test'
# args.saved = './data/split_{}'.format(args.dataset)


# ## == Apply seed ======================
# np.random.seed(args.seed)


# train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
# test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
# X_train, y_train = train_data[:, 1:], train_data[:, 0]
# X_test, y_test = test_data[:, 1:], test_data[:, 0]


# img_view = (1, 28, 28)
# X_test = X_test.reshape((X_test.shape[0], *img_view))
# X_test = torch.tensor(X_test, dtype=torch.float32)

# topil_trans = transforms.ToPILImage()
# totensor_trans = transforms.ToTensor()

# angles = [0, 20, 40, 60, 80]
# for t in range(args.n_tasks):
#   rotated_xtest = transforms.functional.rotate(topil_trans(X_test[1]), angles[t])
#   rotated_xtest = totensor_trans(rotated_xtest)
#   rotated_xtest = (rotated_xtest*255)
#   imshow(torch.stack([X_test[1], rotated_xtest]))
