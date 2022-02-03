import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd

from trainers.episodic_train import train as episodic_train
from trainers.batch_train import train as batch_train
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation


def increm_learn(model,
                  learner,
                  memory,
                  args, device):

  all_tasks_acc_dist = []
  all_tasks_acc_cls = []
  
  for task in range(args.n_tasks):  
    ### === Task data loading =====================
    task_data = pd.read_csv(
                  os.path.join(args.split_train_path, "task_{}.csv".format(task)),
                  sep=',', header=None).values 
    print('task_data: {}'.format(task_data.shape))

    ### === Split setting =========================
    if task != 0:
      if task == 5: args.ways = 20
      replay_mem = memory()
      train_data = np.concatenate((task_data, replay_mem))
      print('replay_mem: {}'.format(replay_mem.shape))
      print('train_data(new): {}'.format(train_data.shape))
      
      ## == Train Model (Batch) ===========
      if args.algorithm == 'batch':
        batch_train(
          model,
          learner,
          train_data,
          args, device)
      ## == Train Model (Episodic) ========
      else:
        episodic_train(
          model,
          learner,
          train_data,
          args, device)
    else:
      ## == Train Model (Batch) ===========
      if args.algorithm == 'batch':
        batch_train(
          model,
          learner,
          task_data,
          args, device)

      ## == Train Model (Episodic) ========
      else:
        episodic_train(
          model,
          learner,
          task_data,
          args, device)
      
    ### == Update memoty =================
    memory.update(task_data)


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


    ### === After each task evaluation =============
    print('=== Testing ... ===')
    if args.which_model == 'best':
      model.load(args.best_model_path)
    
    tasks_acc_dist = [0.0 for _ in range(args.n_tasks)]
    tasks_acc_cls = [0.0 for _ in range(args.n_tasks)]

    for prev_task in range(task+1):
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
      known_labels = set(range((task+1)*2))
      # print('Known_labels for task {} is: {}'.format(task, known_labels))
      _, acc_dis, acc_cls = learner.evaluate(model,
                                            test_dataloader,
                                            known_labels,
                                            args)
      tasks_acc_dist[prev_task] = acc_dis
      tasks_acc_cls[prev_task] = acc_cls
    
    all_tasks_acc_dist.append(torch.tensor(tasks_acc_dist))
    all_tasks_acc_cls.append(torch.tensor(tasks_acc_cls))

    mean_acc_dist = np.mean(tasks_acc_dist[:task+1])
    mean_acc_cls = np.mean(tasks_acc_cls[:task+1])
    print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
    print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
    # print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
    # print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
    print('Mean -> Dist: {}, Cls: {}'.format(round(mean_acc_dist, 3), round(mean_acc_cls, 3)))

  ### == Claculate forgetting =================
  all_tasks_acc_cls = torch.stack(all_tasks_acc_cls)
  all_tasks_acc_dist = torch.stack(all_tasks_acc_dist)

  print(all_tasks_acc_cls)

  acc_cls_best = torch.max(all_tasks_acc_cls, 0).values
  temp = acc_cls_best - all_tasks_acc_cls

  print(temp)

  forgetting_cls = torch.tensor([torch.mean(temp[i+1:, i])  for i in range(args.n_tasks-1)])

  print(forgetting_cls)

  mean_forgetting_cls = torch.mean(forgetting_cls)
  std_forgetting_cls = torch.std(forgetting_cls)
  print('cls forgetting: {:.4f} ± {:.4f}'.format(mean_forgetting_cls, std_forgetting_cls))




  acc_dist_best = torch.max(all_tasks_acc_dist, 0).values
  temp = acc_dist_best - all_tasks_acc_dist
  forgetting_dist = torch.tensor([torch.mean(temp[:, i:])  for i in range(args.n_tasks)])
  mean_forgetting_dist = torch.mean(forgetting_dist[:-1])
  std_forgetting_dist = torch.std(forgetting_dist[:-1])
  print('dist forgetting: {:.4f} ± {:.4f}'.format(mean_forgetting_dist, std_forgetting_dist))


# def batch_increm_learn(model,
#                       memory,
#                       args,
#                       device):
#   print('===== Batch Incremental Learning =========================')  
#   for task in range(args.n_tasks):  
#     print('=== Training ... ===')
#     ### === Task data loading =====================
#     task_data = pd.read_csv(
#                   os.path.join(args.split_train_path, "task_{}.csv".format(task)),
#                   sep=',', header=None).values 
#     print('task_data: {}'.format(task_data.shape))
    
#     ### === Split setting =========================
#     if task != 0:
#       replay_mem = memory()
#       train_data = np.concatenate((task_data, replay_mem))
#       print('replay_mem: {}'.format(replay_mem.shape))
#       print('train_data(new): {}'.format(train_data.shape))
#       batch_train(model,
#         train_data,
#         args, device)
#     else:
#       batch_train(model,
#         task_data,
#         args, device)
#     memory.update(task_data)
    
#     ### === Without Memory ========================
#     # batch_train(model,
#     #   task_data,
#     #   args, device)

#     ### === Drift setting (with memory) ===========
#     # if task == 0:
#     #   batch_train(model,
#     #               task_data,
#     #               args, device)
#     #   memory.update(task_data)
#     # else:
#     #   replay_mem = memory()
#     #   train_data = np.concatenate((task_data, replay_mem))
#     #   batch_train(model,
#     #               train_data,
#     #               args, device)   
    
#     ### === After each task evaluation =============
#     print('=== Testing ... ===')
#     if args.which_model == 'best':
#       model.load(args.best_model_path)
    
#     tasks_acc_cls = [0.0 for _ in range(args.n_tasks)]

#     for prev_task in range(task+1):  
#       test_data = pd.read_csv(
#                   os.path.join(args.split_test_path, "task_{}.csv".format(prev_task)),
#                   sep=',', header=None).values
#       if args.use_transform:
#         _, test_transform = transforms_preparation()
#         test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
#       else:
#         test_dataset = SimpleDataset(test_data, args)
#       test_dataloader = DataLoader(dataset=test_dataset,
#                                    batch_size=args.batch_size,
#                                    shuffle=False)

#       acc_cls = test(model, test_dataloader, args, device)
#       tasks_acc_cls[prev_task] = acc_cls
    
#     mean_acc_cls = np.mean(tasks_acc_cls[:task+1])
#     print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
#     # print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
    
#     print('Mean -> Cls: {}'.format(round(mean_acc_cls, 3)))


# def episodic_increm_learn(model,
#                       learner,
#                       memory,
#                       args,
#                       device):
#   print('===== Episodic Incremental Learning =========================')
#   for task in range(args.n_tasks):  
#     print('=== Training ... ===')
#     ### === Task data loading =====================
#     task_data = pd.read_csv(
#                   os.path.join(args.split_train_path, "task_{}.csv".format(task)),
#                   sep=',', header=None).values 
#     print('task_data: {}'.format(task_data.shape))
    
#     ### === Split setting =========================
#     if task != 0:
#       if task == 5: args.ways = 20
#       replay_mem = memory()
#       train_data = np.concatenate((task_data, replay_mem))
#       print('replay_mem: {}'.format(replay_mem.shape))
#       print('train_data(new): {}'.format(train_data.shape))
#       episodic_train(model,
#         learner,
#         train_data,
#         args, device)
#     else:
#       episodic_train(model,
#         learner,
#         task_data,
#         args, device)
#     memory.update(task_data)
    
#     ### === Without Memory ========================
#     # episodic_train(model,
#     #   learner,
#     #   task_data,
#     #   args, device)

#     ### === Drift setting (with memory) ===========
#     # if task == 0:
#     #   episodic_train(model,
#     #         learner,
#     #         task_data,
#     #         args, device)  
#     #   memory.update(task_data)
#     #   args.beta_type = 'fixed'
#     #   args.beta = args.drift_beta
#     # else:
#     #   replay_mem = memory()
#     #   train_data = np.concatenate((task_data, replay_mem))
#     #   episodic_train(model,
#     #       learner,
#     #       train_data,
#     #       args, device) 

#     ### === After each task evaluation =============
#     print('=== Testing ... ===')
#     if args.which_model == 'best':
#       model.load(args.best_model_path)
    
#     tasks_acc_dist = [0.0 for _ in range(args.n_tasks)]
#     tasks_acc_cls = [0.0 for _ in range(args.n_tasks)]

#     for prev_task in range(task+1):
#       test_data = pd.read_csv(
#                   os.path.join(args.split_test_path, "task_{}.csv".format(prev_task)),
#                   sep=',', header=None).values

#       if args.use_transform:
#         _, test_transform = transforms_preparation()
#         test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
#       else:
#         test_dataset = SimpleDataset(test_data, args)
#       test_dataloader = DataLoader(dataset=test_dataset,
#                                    batch_size=args.batch_size,
#                                    shuffle=False)

#       # known_labels = test_dataset.label_set
#       # print('Test on: {}'.format(known_labels))
#       known_labels = set(range((task+1)*5))
#       # print('Known_labels for task {} is: {}'.format(task, known_labels))
#       _, acc_dis, acc_cls = learner.evaluate(model,
#                                             test_dataloader,
#                                             known_labels,
#                                             args)
#       tasks_acc_dist[prev_task] = acc_dis
#       tasks_acc_cls[prev_task] = acc_cls
    
#     mean_acc_dist = np.mean(tasks_acc_dist[:task+1])
#     mean_acc_cls = np.mean(tasks_acc_cls[:task+1])
#     # print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
#     # print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
#     print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
#     print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
#     print('Mean -> Dist: {}, Cls: {}'.format(round(mean_acc_dist, 3), round(mean_acc_cls, 3)))

    

    


