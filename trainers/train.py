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


from learners.reptile_learner import reptile_learner, reptile_evaluate
from learners.pt_learner import PtLearner
from losses import TotalLoss
from augmentation import transforms
from augmentation.autoaugment.autoaugment import CIFAR10Policy
from utils.functions import imshow


def train(model,
          train_data,
          args,
          device):
  model.to(device)

  train_dataloaders,\
  val_dataloader=  dataloader_preparation(train_data, args)

  ## = Model Update config.
  # criterion  = nn.CrossEntropyLoss()
  # criterion_mt = losses.NTXentLoss(temperature=0.07)
  # criterion = PrototypicalLoss(n_support=args.shot)
  criterion = TotalLoss(args)
  
  optim = SGD(model.parameters(),
              lr=args.lr,
              momentum=args.momentum)
  # optim = Adam(model.parameters(),
  #               lr=args.lr,
  #               weight_decay=args.wd)

  scheduler = StepLR(
    optim,
    step_size=args.step_size,
    gamma=args.gamma,
  )

  ## == 2) Learn model
  global_time = time.time()
  min_loss = float('inf')
 
  pt_learner = PtLearner(criterion, device)

  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('===================================== Epoch %d =====================================' % epoch_item)
      train_loss = 0.
      
      train_loader_iterations = [
        iter(train_loader) for train_loader in train_dataloaders
      ]

      for miteration_item in range(args.meta_iteration):
        queue = [next(trainloader) for trainloader in train_loader_iterations]  
      
        loss = pt_learner.train(model,queue,optim,miteration_item,args)
        train_loss += loss

        ## == validation ==============
        if (miteration_item + 1) % args.log_interval == 0:
          
          train_loss_total = train_loss / args.log_interval
          train_loss = 0.

          # evalute on val_dataset
          # val_loss_total = reptile_evaluate(model, val_dataloader, criterion, device) # For Reptile
          val_loss_total = pt_learner.evaluate(model, val_dataloader)  # For Pt.
          
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
    
      if args.scheduler:
        scheduler.step()

  except KeyboardInterrupt:
    print('skipping training')  
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  print("Saving new last model")




if __name__ == '__main__':
  pass




