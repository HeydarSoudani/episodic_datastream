import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import os
import time
import numpy as np
import pandas as pd

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
# from phases.incremental_learn import increm_test


def increm_test(model, learner, current_task, args):
  
  tasks_acc_dist = [0.0 for _ in range(args.n_tasks)]
  tasks_acc_cls = [0.0 for _ in range(args.n_tasks)]
  
  for prev_task in range(current_task+1):
    test_data = pd.read_csv(
                os.path.join(args.split_test_path, "task_{}.csv".format(prev_task)),
                sep=',', header=None).values

    if args.use_transform:
      _, test_transform = transforms_preparation()
      test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
    else:
      test_dataset = SimpleDataset(test_data, args)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)

    # known_labels = test_dataset.label_set
    # print('Test on: {}'.format(known_labels))
    if args.dataset == 'cifar100':
      known_labels = set(range((current_task+1)*5))
    else:
      known_labels = set(range((current_task+1)*2))
    # print('Known_labels for task {} is: {}'.format(task, known_labels))
    
    _, acc_dis, acc_cls = learner.evaluate(model,
                                          test_dataloader,
                                          known_labels,
                                          args)
                 
    tasks_acc_dist[prev_task] = acc_dis
    tasks_acc_cls[prev_task] = acc_cls
  
  return tasks_acc_dist, tasks_acc_cls


def train(model,
          learner,
          train_data,
          args, device,
          current_task
          ):              
  model.to(device)


  # == For trajectory ===
  all_dist_acc = {'task_{}'.format(i): [] for i in range(args.n_tasks)}
  all_cls_acc = {'task_{}'.format(i): [] for i in range(args.n_tasks)}


  ### === Load data ======================
  n, _ = train_data.shape
  np.random.shuffle(train_data)
  train_val_data = np.split(train_data, [int(n*0.95), n])
  train_data = train_val_data[0]
  val_data = train_val_data[1]

  train_transform, test_transform = transforms_preparation()
  if args.use_transform:
    train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
    val_dataset = SimpleDataset(val_data, args, transforms=test_transform)
  else:
    train_dataset = SimpleDataset(train_data, args)
    val_dataset = SimpleDataset(val_data, args)

  train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
  val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

  known_labels = val_dataset.label_set

  # == ====================================
  optim = Adam(model.parameters(), lr=args.lr)
  # optim = SGD(model.parameters(),
  #             lr=args.lr,
  #             momentum=args.momentum)
  scheduler = StepLR(optim, step_size=args.step_size, gamma=args.gamma)

  # == Learn model =========================
  global_time = time.time()
  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch {} ==='.format(epoch_item+1))
      train_loss = 0.
      for i, batch in enumerate(train_loader):
        
        loss = learner.train(model, batch, optim, args)
        train_loss += loss

        # evaluation
        if (i+1) % args.log_interval == 0:
          with torch.no_grad():
            
            train_loss_total = train_loss / args.log_interval
            train_loss = 0.0
            
            # evalute on val_dataset
            val_loss_total, \
            val_acc_dis_total, \
            val_acc_cls_total = learner.evaluate(model, val_loader, known_labels, args)

            print('=== Time: %.2f, Epoch: %d/%d, Train Loss: %f, Val Loss: %f' % (
              time.time()-global_time, epoch_item+1, i+1, train_loss_total, val_loss_total))
            global_time = time.time()

            # save best model
            if val_loss_total < min_loss:
              model.save(os.path.join(args.save, "model_best.pt"))
              min_loss = val_loss_total
              print("Saving new best model")
            
            # =====================
            # == For trajectory ===
            print('Prototypes are calculating ...')
            learner.calculate_prototypes(model, train_loader)
            
            tasks_acc_dist, tasks_acc_cls = increm_test(model, learner, current_task, args)
            for j in range(current_task+1):
              all_dist_acc['task_{}'.format(j)].append(round(tasks_acc_dist[j]*100, 2))
              all_cls_acc['task_{}'.format(j)].append(round(tasks_acc_cls[j]*100, 2))
            # =====================

        if args.scheduler:
          scheduler.step()

  except KeyboardInterrupt:
    print('skipping training')  
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  print("Saving new last model")

  # Claculate Pts.
  print('Prototypes are calculating ...')
  learner.calculate_prototypes(model, train_loader)

  # save learner
  learner.save(os.path.join(args.save, "learner.pt"))
  print("= ...Learner saved")


  # == For trajectory ===
  return all_dist_acc, all_cls_acc

