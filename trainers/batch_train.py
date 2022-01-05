import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np

from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation

def train(model,
          train_data,
          args, device):              
  model.to(device)

  ### === Load data =================
  n, _ = train_data.shape
  np.random.shuffle(train_data)
  train_val_data = np.split(train_data, [int(n*0.9), n])
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


  criterion = torch.nn.CrossEntropyLoss()
  optim = SGD(model.parameters(),
              lr=args.lr,
              momentum=args.momentum)
  # optim = SGD(model.parameters(),
  #             lr=args.lr,
  #             momentum=args.momentum,
  #             weight_decay=args.wd,  #l2 reg
  #             nesterov=True)
  # optim = Adam(model.parameters(),
  #               lr=args.lr,
  #               weight_decay=args.wd)
  # scheduler = StepLR(optim, step_size=8, gamma=args.gamma)
  
  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch %d ===' % epoch_item)
      train_loss = 0.
      for i, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optim.zero_grad()
        outputs, _ = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        train_loss += loss

        if (i+1) % args.log_interval == 0:
          with torch.no_grad():
            total_val_loss = 0.0
            model.eval()
            for j, data in enumerate(val_loader):
              sample, labels = data
              sample, labels = sample.to(device), labels.to(device)

              logits, _ = model.forward(sample)
    
              loss = criterion(logits, labels)
              loss = loss.mean()
              total_val_loss += loss.item()

            total_val_loss /= len(val_loader)
            print('=== Epoch: %d/%d, Train Loss: %f, Val Loss: %f' % (
              epoch_item, i+1,  train_loss/args.log_interval, total_val_loss))
            train_loss = 0.

            # save best model
            if total_val_loss < min_loss:
              model.save(os.path.join(args.save, "model_best.pt"))
              min_loss = total_val_loss
              print("Saving new best model")

      # scheduler.step()

  except KeyboardInterrupt:
    print('skipping training')  
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  print("Saving new last model")


def test(model, test_loader, args, device):
  model.to(device)

  correct = 0
  total = 0

  model.eval()
  with torch.no_grad():
    for i, data in enumerate(test_loader):
  
      samples, labels = data
      samples, labels = samples.to(device), labels.to(device)
      logits, feature = model.forward(samples)
      
      _, predicted = torch.max(logits, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    
    # print('Accuracy of the network on the 10000 test images: %7.4f %%' % (100 * correct / total))
  return correct / total
  

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


