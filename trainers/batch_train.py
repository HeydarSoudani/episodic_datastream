import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import os
import numpy as np

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation

def train(model,
          learner,
          train_data,
          args, device):              
  model.to(device)

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
  val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

  # == ====================================
  optim = Adam(model.parameters(), lr=args.lr)
  # optim = SGD(model.parameters(),
  #             lr=args.lr,
  #             momentum=args.momentum,
  #             weight_decay=args.wd) #l2 reg
  # optim = Adam(model.parameters(),
  #               lr=args.lr,
  #               weight_decay=args.wd)
  scheduler = StepLR(optim, step_size=args.step_size, gamma=args.gamma)
  # scheduler = OneCycleLR(optim, 0.01,
  #                       epochs=args.epochs, 
  #                       steps_per_epoch=len(train_loader))

  # == Learn model =========================
  global_time = time.time()
  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch %d ===' % epoch_item)
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
            val_acc_cls_total = learner.evaluate(model, val_dataloader, args)

            print('=== Time: %.2f, Step: %d, Train Loss: %f, Val Loss: %f' % (
              time.time()-global_time, miteration_item+1, train_loss_total, val_loss_total))
            global_time = time.time()

            # save best model
            if val_loss_total < min_loss:
              model.save(os.path.join(args.save, "model_best.pt"))
              min_loss = total_val_loss
              print("Saving new best model")

        if args.scheduler:
          scheduler.step()

  except KeyboardInterrupt:
    print('skipping training')  
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  print("Saving new last model")








# def test(model, test_loader, args, device):
#   model.to(device)

#   correct = 0
#   total = 0
#   model.eval()
#   with torch.no_grad():
#     for i, data in enumerate(test_loader):
#       samples, labels = data
#       samples, labels = samples.to(device), labels.to(device)
#       logits, feature = model.forward(samples)
      
#       _, predicted = torch.max(logits, 1)
#       total += labels.size(0)
#       correct += (predicted == labels).sum().item()
    
#     print('Accuracy of the network on the 10000 test images: %7.4f %%' % (100 * correct / total))
#   return correct / total
  

  # prepare to count predictions for each class
  # classes = ('tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot')

  # correct_pred = {classname: 0 for classname in classes}
  # total_pred = {classname: 0 for classname in classes}

  # # again no gradients needed
  # with torch.no_grad():
  #   for data in test_dataloader:
  #     sample, labels = data
  #     sample, labels = sample.to(device), labels.to(device)
  #     out, feature = model.forward(sample)
  #     _, predictions = torch.max(out.data, 1)
  #     # collect the correct predictions for each class
  #     for label, prediction in zip(labels, predictions):
  #       if label == prediction:
  #         correct_pred[classes[label]] += 1
  #       total_pred[classes[label]] += 1


  # # print accuracy for each class
  # for classname, correct_count in correct_pred.items():
  #   accuracy = 100 * float(correct_count) / total_pred[classname]
  #   print("Accuracy for class {:11s} is: {:.1f} %".format(classname, accuracy))


