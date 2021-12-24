from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd

# from trainers.episodic_train import train
from trainers.batch_train import train, test
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation


def increm_learn(model,
                 learner,
                 memory,
                 args,
                 device):
  print('================================ Incremental Learning =========================')
  # f = open('inc_output.txt','w')
  
  for task in range(args.n_tasks):  
    print('=== Training ... ===')
    ## = Data ===========
    task_data = pd.read_csv(
                  os.path.join(args.split_train_path, "task_{}.csv".format(task)),
                  sep=',', header=None).values 
    print('task_data: {}'.format(task_data.shape))

    # if task != 0:
    #   if task == 2: args.ways = 5
    #   replay_mem = memory()
    #   train_data = np.concatenate((task_data, replay_mem))
    #   print('replay_mem: {}'.format(replay_mem.shape))
    #   print('train_data(new): {}'.format(train_data.shape))

    #   ### === train ==============
    #   # 1) episodic training
    #   train(model,
    #         learner,
    #         train_data,
    #         args, device)
      # 2) batch training
    # train(model,
    #       train_data,
    #       args, device)
    # else:
      # 1) episodic training
    # train(model,
    #       learner,
    #       task_data,
    #       args, device)
      # 2) batch training
      #    
    
    train(model,
        task_data,
        args, device)
    
    if task == 0:
      # train(model,
      #   task_data,
      #   args, device)
      # train(model,
      #       learner,
      #       task_data,
      #       args, device)  
      # = Update memory =====
      # memory.update(task_data)
      args.beta_type = 'fixed'
      args.beta = 0.999
    # else:
    #   replay_mem = memory()
    #   train_data = np.concatenate((task_data, replay_mem))
    #   train(model,
    #       train_data,
    #       args, device)   
      # train(model,
      #       learner,
      #       train_data,
      #       args, device) 

    # = evaluation ========
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

      ### 1) episodic test
      # known_labels = test_dataset.label_set
      # # #known_labels = set(range((task+1)*2))
      # _, acc_dis, acc_cls = learner.evaluate(model,
      #                                       test_dataloader,
      #                                       known_labels,
      #                                       args)
      ### 2) batch test
      acc_cls = test(model, test_dataloader, args, device)

      # tasks_acc_dist[prev_task] = acc_dis
      tasks_acc_cls[prev_task] = acc_cls
    
    # mean_acc_dist = np.mean(tasks_acc_dist[:task+1])
    mean_acc_cls = np.mean(tasks_acc_cls[:task+1])
    
    # print("Dist acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_dist))
    print("Cls  acc.: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(tasks_acc_cls))
    # print('Mean -> Dist: {}, Cls: {}'.format(round(mean_acc_dist, 3), round(mean_acc_cls, 3)))
    print('Mean -> Cls: {}'.format(round(mean_acc_cls, 3)))
    # f.write("%7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(prev_tasks_acc))
    

    


