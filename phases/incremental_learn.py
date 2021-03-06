import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import time

from trainers.episodic_train import train as episodic_train
from trainers.batch_train import train as batch_train
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation


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


def increm_learn(model,
                  learner,
                  memory,
                  args, device):

  # -- For acc. trajectory ----
  traj_dist_acc = {'task_{}'.format(i): [] for i in range(args.n_tasks)}
  traj_cls_acc = {'task_{}'.format(i): [] for i in range(args.n_tasks)}

  # -- For forgetting ----
  all_tasks_acc_dist = []
  all_tasks_acc_cls = []
  
  # -- Time lists ---------
  train_times = []
  eval_times = []
  memory_times = []
  
  for task in range(args.n_tasks):  
    ### === Task data loading =====================
    task_data = pd.read_csv(
                  os.path.join(args.split_train_path, "task_{}.csv".format(task)),
                  sep=',', header=None).values 
    print('task_data: {}'.format(task_data.shape))

    train_start_time = time.time()
    if task != 0:
      
      ### === Split setting ===============
      if args.dataset == 'cifar100':
        if task == 5: args.ways = 20
      else:
        if task == 1: args.ways = 4

      replay_mem = memory()
      train_data = np.concatenate((task_data, replay_mem))
      print('replay_mem: {}'.format(replay_mem.shape))
      print('train_data(new): {}'.format(train_data.shape))
      
      ## == Train Model (Batch) ===========
      # -- Output model for trajectory ----
      if args.algorithm == 'batch':
        local_dist_acc, local_cls_acc = batch_train(
          model,
          learner,
          train_data,
          args, device,
          current_task=task) # current_task parameter for acc. trajectory
        # batch_train(
        #   model,
        #   learner,
        #   train_data,
        #   args, device)
      ## == Train Model (Episodic) ========
      else:
        local_dist_acc, local_cls_acc = episodic_train(
          model,
          learner,
          train_data,
          args, device,
          current_task=task) # current_task parameter for acc. trajectory
        # episodic_train(
        #   model,
        #   learner,
        #   train_data,
        #   args, device)
    else:
      ## == Train Model (Batch) ===========
      if args.algorithm == 'batch':
        local_dist_acc, local_cls_acc = batch_train(
          model,
          learner,
          task_data,
          args, device,
          current_task=task)
        # batch_train(
        #   model,
        #   learner,
        #   task_data,
        #   args, device)

      ## == Train Model (Episodic) ========
      else:
        local_dist_acc, local_cls_acc = episodic_train(
          model,
          learner,
          task_data,
          args, device,
          current_task=task) # current_task parameter for acc. trajectory
        # episodic_train(
        #   model,
        #   learner,
        #   task_data,
        #   args, device)
    train_times.append(time.time() - train_start_time)
    
    ### == Update memoty ===================
    mem_start_time = time.time()
    memory.update(task_data)
    memory_times.append(time.time() - mem_start_time)

    # == For acc. trajectory ============
    for i in range(task+1):
      traj_dist_acc['task_{}'.format(i)].extend(local_dist_acc['task_{}'.format(i)])
      traj_cls_acc['task_{}'.format(i)].extend(local_cls_acc['task_{}'.format(i)])
    print(traj_dist_acc)
    print(traj_cls_acc)

    ### === After each task evaluation =====
    print('=== Testing ... ===')
    if args.which_model == 'best':
      model.load(args.best_model_path)

    eval_start_time = time.time()
    tasks_acc_dist, tasks_acc_cls = increm_test(model, learner, task, args)
    eval_times.append(time.time() - eval_start_time)    

    all_tasks_acc_dist.append(torch.tensor(tasks_acc_dist))
    all_tasks_acc_cls.append(torch.tensor(tasks_acc_cls))

    mean_acc_dist = np.mean(tasks_acc_dist[:task+1])
    mean_acc_cls = np.mean(tasks_acc_cls[:task+1])
    
    ## == Print results ==========
    if args.dataset == 'cifar100': 
      print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
      print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
    else:
      print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
      print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
    print('Mean -> Dist: {}, Cls: {}'.format(round(mean_acc_dist, 3), round(mean_acc_cls, 3)))

  ### == Claculate forgetting =================
  all_tasks_acc_cls = torch.stack(all_tasks_acc_cls)
  all_tasks_acc_dist = torch.stack(all_tasks_acc_dist)

  acc_dist_best = torch.max(all_tasks_acc_dist, 0).values
  temp = acc_dist_best - all_tasks_acc_dist
  forgetting_dist = torch.tensor([torch.mean(temp[i+1:, i]) for i in range(args.n_tasks-1)])
  mean_forgetting_dist = torch.mean(forgetting_dist)
  std_forgetting_dist = torch.std(forgetting_dist)
  print('dist forgetting: {:.4f} ?? {:.4f}'.format(mean_forgetting_dist, std_forgetting_dist))

  acc_cls_best = torch.max(all_tasks_acc_cls, 0).values
  temp = acc_cls_best - all_tasks_acc_cls
  forgetting_cls = torch.tensor([torch.mean(temp[i+1:, i]) for i in range(args.n_tasks-1)])
  mean_forgetting_cls = torch.mean(forgetting_cls)
  std_forgetting_cls = torch.std(forgetting_cls)
  print('cls forgetting: {:.4f} ?? {:.4f}'.format(mean_forgetting_cls, std_forgetting_cls))

  ## == Print time =============================
  all_time = sum(train_times)+sum(memory_times)+sum(eval_times)
  print("Time: %7.4f, %7.4f, %7.4f, %7.4f"%
      (sum(train_times), sum(memory_times), sum(eval_times), all_time ))








    # tasks_acc_dist = [0.0 for _ in range(args.n_tasks)]
    # tasks_acc_cls = [0.0 for _ in range(args.n_tasks)]

    # for prev_task in range(task+1):
    #   test_data = pd.read_csv(
    #               os.path.join(args.split_test_path, "task_{}.csv".format(prev_task)),
    #               sep=',', header=None).values

    #   if args.use_transform:
    #     _, test_transform = transforms_preparation()
    #     test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
    #   else:
    #     test_dataset = SimpleDataset(test_data, args)
    #   test_dataloader = DataLoader(dataset=test_dataset,
    #                                batch_size=args.batch_size,
    #                                shuffle=False)

    #   # known_labels = test_dataset.label_set
    #   # print('Test on: {}'.format(known_labels))
    #   if args.dataset == 'cifar100':
    #     known_labels = set(range((task+1)*5))
    #   else:
    #     known_labels = set(range((task+1)*2))
    #   # print('Known_labels for task {} is: {}'.format(task, known_labels))
      
    #   _, acc_dis, acc_cls = learner.evaluate(model,
    #                                         test_dataloader,
    #                                         known_labels,
    #                                         args)
    #   tasks_acc_dist[prev_task] = acc_dis
    #   tasks_acc_cls[prev_task] = acc_cls
    
    # all_tasks_acc_dist.append(torch.tensor(tasks_acc_dist))
    # all_tasks_acc_cls.append(torch.tensor(tasks_acc_cls))



    ### === Without Memory ========================
    ## == Train Model (Batch) ===========
    # if args.algorithm == 'batch':
    #   batch_train(
    #     model,
    #     learner,
    #     task_data,
    #     args, device)

    # ## == Train Model (Episodic) ========
    # else:
    #   episodic_train(
    #     model,
    #     learner,
    #     task_data,
    #     args, device)

    ### === Drift setting (with memory) ===========
    # if task == 0:
    #   episodic_train(model,
    #         learner,
    #         task_data,
    #         args, device)  
    #   memory.update(task_data)
    #   args.beta_type = 'fixed'
    #   args.beta = args.drift_beta
    # else:
    #   replay_mem = memory()
    #   train_data = np.concatenate((task_data, replay_mem))
    #   episodic_train(model,
    #       learner,
    #       train_data,
    #       args, device) 