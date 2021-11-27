import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np

from trainers.train import train
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation


def evaluate(model, dataloader, device):
  ce = torch.nn.CrossEntropyLoss()
  correct = 0
  total = 0
  
  model.eval()
  with torch.no_grad():
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
      sample, labels = batch
      sample, labels = sample.to(device), labels.to(device)
      
      logits, _ = model.forward(sample)
      
      _, predicted = torch.max(logits, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      
      loss = ce(logits, labels)
      loss = loss.mean()
      total_loss += loss.item()

  # print('correct: {}'.format(correct))
  # print('total: {}'.format(total))
  acc = 100 * correct / total  
  total_loss /= len(dataloader)
  return acc, total_loss


def increm_learn(model,
                 pt_learner,
                 memory,
                 args,
                 device):
  print('================================ Incremental Learning =========================')
  f = open('inc_output.txt','w')
  
  for task in range(args.n_tasks):  
    print('=== Training ... ===')
    ## = Data ===========
    train_data = pd.read_csv(
                  os.path.join(args.split_train_path, "task_{}.csv".format(task)),
                  sep=',', header=None).values
    
    print('train data: {}'.format(train_data.shape))

    if task != 0:
      args.ways += 1 
      replay_mem = memory()
      new_train_data = np.concatenate((train_data, replay_mem))
      print('replay_mem: {}'.format(replay_mem.shape))
      print('train_data (new): {}'.format(new_train_data.shape))

      # = train ===========
      train(model, pt_learner, new_train_data, args, device)
    
    else:
      train(model, pt_learner, train_data, args, device)
    
    # = Update memory =====
    # memory.update(pt_learner.prototypes)
    memory.update(train_data)
    
    # = evaluation ========
    print('=== Testing ... ===')
    # 1) average performance on all the samples regardless of their task.
    # 2) average performance up till current task.

    if args.which_model == 'best':
      model.load(args.best_model_path)
    
    prev_tasks_acc = [0.0 for _ in range(args.n_tasks)]
    for prev_task in range(task+1):
      
      test_data = pd.read_csv(
                  os.path.join(args.split_test_path, "task_{}.csv".format(prev_task)),
                  sep=',', header=None).values

      if args.use_transform:
        _, test_transform = transforms_preparation()
        test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
      else:
        test_dataset = SimpleDataset(test_data, args)
      test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

      acc, _ = evaluate(model, test_dataloader, device)
      prev_tasks_acc[prev_task] = acc

    print("%7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(prev_tasks_acc))
    f.write("%7.4f, %7.4f, %7.4f, %7.4f, %7.4f \n"% tuple(prev_tasks_acc))
    

    


