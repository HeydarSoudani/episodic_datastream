# import os
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# from datasets.dataset import SimpleDataset


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
#     'rfmnist',
#     'cifar10'
#   ],
#   default='rfmnist',
#   help='')
# parser.add_argument('--seed', type=int, default=5, help='')
# args = parser.parse_args()

# # = Add some variables to args ========
# if args.dataset in ['mnist', 'pmnist', 'rmnist']:
#   data_folder = 'mnist'
# elif args.dataset in ['fmnist', 'pfmnist', 'rfmnist']:
#   data_folder = 'fmnist'
# else:
#   data_folder = args.dataset

# args.data_path = 'data/{}'.format(data_folder)
# args.train_path = 'train'
# args.test_path = 'test'
# args.saved = './data/split_{}'.format(args.dataset)

# ## == Apply seed ======================
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)


# def imshow(imgs):
#   imgs *= 255.5
#   grid_imgs = torchvision.utils.make_grid(torch.tensor(imgs), nrow=2)
#   plt.imshow(grid_imgs.permute(1, 2, 0))
#   plt.show()


# def show_samples():
  
#   ## incremental version
#   for task in range(args.n_tasks):
#     ## = Data ===========
#     task_data = pd.read_csv(
#       os.path.join(args.split_train_path, "task_{}.csv".format(task)),
#       sep=',', header=None).values 
#     print('task_data: {}'.format(task_data.shape))
#     dataset = SimpleDataset(task_data, args)
#     sampler = PtSampler(
#       dataset,
#       n_way=10,
#       n_shot=1,
#       n_query=0,
#       n_tasks=1)
#     dataloader = DataLoader(
#       test_dataset,
#       batch_sampler=sampler,
#       num_workers=1,
#       pin_memory=True,
#       collate_fn=sampler.episodic_collate_fn)

#     batch = next(iter(test_dataloader))
#     support_images, support_labels, _, _ = batch

#     imshow(support_images)

# if __name__ == '__main__':
#   show_samples()
