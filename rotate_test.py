# import torch
# import torchvision
# import torchvision.transforms as transforms

# from scipy import ndimage
# import matplotlib.pyplot as plt
# import pandas as pd 
import numpy as np
# import argparse
# import os


# a = [0.7949, 0.8101, 0.7268, 0.8114, 0.7370, 0.8248]
a = [91.43, 84.62, 83.26, 81.76, 78.42]

print(np.mean(a))
print(np.std(a))

f = 0.0
for item in a:
  f += a[0] - item
f /= (len(a)-1)
print(f)




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
