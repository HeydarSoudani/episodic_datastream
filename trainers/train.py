import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os
import time
import numpy as np
# from pytorch_metric_learning import distances, losses, miners
from collections import Counter

from torchvision.transforms.transforms import RandomRotation

from utils.preparation import dataloader_preparation

from generator import task_generator
from sampler import TaskSampler
from datasets.dataset import DatasetFM
from learners.reptile_learner import reptile_learner, reptile_evaluate
from learners.pt_learner import pt_learner, pt_evaluate
from losses import CPELoss
from augmentation import transforms
from augmentation.autoaugment.autoaugment import CIFAR10Policy
from utils.functions import imshow


def train(model,
          train_data,
          args,
          device):
  # class_weights=[1., 1., 1.]
  model.to(device)

  train_dataloaders, val_dataloader=  dataloader_preparation(train_data, args)
  ### === Seperate data to train and val set ===============
  print(train_data.shape)
  n = train_data.shape[0]
  np.random.shuffle(train_data)
  train_val_data = np.split(train_data, [int(n*0.9), n])
  train_data = train_val_data[0]
  val_data = train_val_data[1]

  ## == 1) Create tasks from train_data
  # task: [n, 765(feature_in_line+label)]
  task_list = task_generator(train_data, args, task_number=1, type='random') #['random', 'dreca']

  ## =========
  transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4, fill=128),
    # transforms.RandomHorizontalFlip(p=0.5),
    # CIFAR10Policy(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])
  transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])


  # loader
  train_loaders = []
  for task_data in task_list:
    # temp_dataset = DatasetFM(task_data)
    temp_dataset = DatasetFM(task_data, transforms=transform_train)

    train_sampler = TaskSampler(
      temp_dataset,
      n_way=args.ways,
      # n_query_way=args.query_ways,
      n_shot=args.shot,
      n_query=args.query_num,
      n_tasks=args.meta_iteration
    )
    train_loader = DataLoader(
      temp_dataset,
      batch_sampler=train_sampler,
      num_workers=1,
      pin_memory=True,
      collate_fn=train_sampler.episodic_collate_fn,
    )
    train_loaders.append(train_loader)

  ## = Data config.
  val_dataset = DatasetFM(val_data)
  # val_dataset = DatasetFM(val_data, transforms=transform_val)
  val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)


  ## = Model Update config.
  # criterion  = nn.CrossEntropyLoss()
  # criterion_mt = losses.NTXentLoss(temperature=0.07)
  criterion = CPELoss(args)
  # criterion = PrototypicalLoss(n_support=args.shot)
  optim = SGD(model.parameters(),
              lr=args.lr,
              momentum=args.momentum)
  # optim = Adam(model.parameters(),
  #               lr=args.lr,
  #               weight_decay=args.wd)

  # scheduler = StepLR(optim, step_size=2, gamma=args.gamma)

  ## == 2) Learn model
  global_time = time.time()
  min_loss = float('inf')
 
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('===================================== Epoch %d =====================================' % epoch_item)
      train_loss = 0.
      
      # for miteration_item in range(args.meta_iteration): 
        
      for train_loader in train_loaders:
        for miteration_item, batch in enumerate(train_loader):
          # batch = next(iter(train_loader))
          
          # == Data preparation ===========
          support_images, support_labels, query_images, query_labels = batch

          support_images = support_images.reshape(1, -1,
                                support_images.size(2),
                                support_images.size(3),
                                support_images.size(4)).squeeze(dim=0)
          support_labels = support_labels.flatten()
          query_images = query_images.reshape(1, -1,
                                query_images.size(2),
                                query_images.size(3),
                                query_images.size(4)).squeeze(dim=0)
          query_labels = query_labels.flatten()

          support_images = support_images.to(device)
          support_labels = support_labels.to(device)
          query_images = query_images.to(device)
          query_labels = query_labels.to(device)

          # imshow(support_images)


          ## == train ===================
          # loss = reptile_learner(model,
          #                       support_images,
          #                       support_labels,
          #                       query_images,
          #                       query_labels,
          #                       criterion, criterion_mt,
          #                       optim,
          #                       args)
          loss, prototypes = pt_learner(model,
                                        support_images,
                                        support_labels,
                                        query_images,
                                        query_labels,
                                        criterion,
                                        optim,
                                        args)
          train_loss += loss

          ## == validation ==============
          if (miteration_item + 1) % args.log_interval == 0:
            
            train_loss_total = train_loss / args.log_interval
            train_loss = 0.

            # evalute on val_dataset
            # val_loss_total = reptile_evaluate(model, val_dataloader, criterion, device) # For Reptile
            val_loss_total = pt_evaluate(model, val_dataloader, prototypes, criterion, device)  # For Pt.
            
            # print losses
            print('Time: %f, Step: %d, Train Loss: %f, Val Loss: %f' % (
              time.time()-global_time, miteration_item+1, train_loss_total, val_loss_total))
            print('===============================================')
            global_time = time.time()
      
            # save best model
            if val_loss_total < min_loss:
              model.save(os.path.join(args.save, "model_best.pt"))
              min_loss = val_loss_total
              print("Saving new best model")

      # scheduler.step()

  except KeyboardInterrupt:
    print('skipping training')  
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  print("Saving new last model")




if __name__ == '__main__':
  pass




