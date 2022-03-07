# import torch
# import torchvision
# import torchvision.transforms as transforms
# from scipy import ndimage
# import matplotlib.pyplot as plt
# import pandas as pd 
import numpy as np
# import argparse
# import os

a = [
  0.5956,
  0.5424,
  0.5797,
  0.5532
]
print('{:.2f} ± {:.2f}'.format(np.mean(a)*100, np.std(a)*100))

### -- Forgetting ----------------
# f = 0.0
# for item in a:
#   f += a[0] - item
# f /= (len(a)-1)
# print(f)



b = [
  {
    'task_0': [99.27, 99.18, 99.64, 99.73, 99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.64, 99.68, 99.64, 99.82, 99.82, 99.82, 99.91, 99.82, 99.91, 99.91, 99.82, 99.91, 99.91, 99.91, 99.82, 99.91, 99.91, 99.82, 87.01, 93.6, 96.02, 96.88, 96.66, 96.8, 97.71, 97.3, 97.66, 97.48, 96.74, 97.25, 97.3, 97.29, 97.11, 97.29, 96.61, 96.89, 96.61, 97.02, 97.2, 96.57, 96.42, 97.25, 96.98, 96.93, 96.79, 96.93, 96.19, 96.01, 81.97, 87.11, 87.57, 89.96, 91.51, 91.47, 92.11, 92.66, 93.3, 92.71, 93.67, 93.82, 94.09, 93.53, 94.73, 93.58, 93.4, 93.86, 92.52, 94.0, 93.34, 93.67, 93.85, 94.36, 94.31, 93.94, 93.85, 92.81, 94.41, 94.68, 87.71, 85.27, 83.02, 84.67, 83.71, 85.59, 86.69, 87.98, 86.51, 86.92, 86.1, 85.49, 87.15, 87.42, 87.98, 88.39, 89.03, 88.94, 90.83, 89.95, 89.26, 91.56, 91.93, 91.01, 89.63, 90.92, 91.06, 92.11, 91.1, 92.99, 85.49, 81.79, 82.29, 82.47, 82.48, 85.32, 86.47, 86.48, 86.52, 86.7, 87.21, 89.09, 88.45, 88.4, 88.9, 87.76, 88.17, 88.86, 89.82, 88.95, 88.49, 89.55, 89.5, 89.05, 89.27, 89.4, 88.9, 89.22, 88.36, 87.53],
    'task_1': [84.86, 95.58, 97.1, 97.44, 98.28, 98.57, 98.62, 98.97, 99.01, 99.01, 99.16, 99.31, 99.4, 99.31, 99.31, 99.26, 99.41, 99.31, 99.56, 99.56, 99.61, 99.6, 99.46, 99.51, 99.41, 99.6, 99.65, 99.56, 99.66, 99.46, 69.38, 76.93, 78.16, 80.8, 80.69, 84.26, 83.91, 86.51, 87.19, 88.46, 88.16, 89.14, 89.15, 89.45, 89.16, 89.83, 88.61, 90.87, 90.03, 90.38, 91.16, 89.45, 89.99, 90.28, 90.32, 91.0, 91.46, 91.54, 92.09, 90.54, 75.73, 78.13, 77.59, 78.46, 79.04, 79.49, 80.52, 80.76, 82.19, 80.29, 80.92, 81.51, 79.94, 81.17, 82.16, 81.46, 82.78, 82.98, 82.59, 81.95, 83.38, 83.57, 83.48, 82.55, 82.5, 82.89, 83.82, 82.99, 83.62, 84.7, 71.96, 73.57, 72.32, 75.21, 77.17, 78.09, 79.16, 79.51, 80.14, 80.25, 80.54, 79.6, 79.9, 80.04, 80.24, 80.64, 81.17, 81.41, 81.36, 81.51, 80.63, 82.24, 81.46, 81.66, 82.24, 80.72, 81.27, 80.83, 80.73, 82.1],
    'task_2': [85.88, 90.62, 93.01, 95.58, 95.99, 96.89, 96.91, 97.3, 97.22, 98.09, 98.72, 98.3, 98.8, 98.45, 98.19, 98.4, 98.35, 98.24, 98.48, 98.45, 98.59, 98.74, 98.63, 98.06, 98.32, 98.45, 99.0, 98.79, 98.77, 98.35, 70.61, 77.36, 77.7, 81.42, 81.83, 81.49, 80.53, 81.57, 80.92, 80.76, 81.08, 81.34, 82.85, 82.51, 84.37, 82.48, 84.69, 87.11, 86.14, 86.63, 86.38, 88.13, 87.07, 86.61, 85.39, 86.58, 87.24, 86.82, 88.68, 89.7, 71.98, 73.04, 71.03, 72.91, 75.21, 74.95, 77.13, 77.65, 77.79, 76.84, 76.79, 77.36, 78.39, 78.16, 78.11, 78.82, 79.23, 77.62, 77.87, 77.01, 76.63, 77.7, 78.35, 78.63, 77.16, 76.78, 77.52, 78.56, 77.26, 77.25],
    'task_3': [92.53, 94.98, 96.05, 95.12, 94.78, 95.02, 94.63, 94.48, 95.12, 94.97, 95.26, 95.31, 95.75, 95.9, 95.9, 95.8, 96.0, 95.56, 95.9, 95.9, 96.04, 95.95, 96.14, 95.95, 95.95, 95.65, 95.85, 95.7, 95.21, 96.09, 77.69, 76.28, 79.59, 80.65, 82.64, 84.39, 86.05, 86.0, 86.97, 87.41, 87.8, 87.22, 86.98, 87.8, 88.04, 88.53, 88.49, 88.82, 88.49, 88.78, 89.12, 89.07, 88.83, 88.39, 87.91, 88.05, 87.62, 87.37, 87.86, 87.76],
    'task_4': [92.28, 95.21, 95.6, 96.33, 96.87, 97.31, 97.21, 97.65, 97.7, 97.85, 98.05, 97.36, 98.24, 98.54, 98.58, 98.63, 98.54, 98.49, 98.54, 98.63, 98.49, 98.93, 98.73, 98.93, 98.97, 98.68, 98.78, 98.97, 99.02, 99.17]
  },
  {

  }
]


output = {
  'mean': {},
  'std': {}
}

for task in range(n_task):
  temp = [item['task_{}'.format(task)]  for item in b]
  output['mean']['task_{}'.format(task)] = np.mean(temp, axis=0)
  output['std']['task_{}'.format(task)] = np.std(temp, axis=0)



































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
