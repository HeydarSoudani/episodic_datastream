import torch
import torchvision
# import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os
import time
import numpy as np
from pandas import read_csv
# from pytorch_metric_learning import distances, losses, miners

from datasets.dataset import DatasetFM
from augmentation import transforms
from utils.functions import imshow, mean_std_calculator
from augmentation.autoaugment.autoaugment import CIFAR10Policy
from augmentation.autoaugment.cutout import Cutout

def batch_train(model, args, device):
  model.to(device)

  dataset_split =  [45000, 5000] #fmnist:[54000, 6000], cifar10: [45000, 5000]

  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(p=0.5),
    CIFAR10Policy(),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])

  batch_size = 64

  # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
  #                                         download=True, transform=transform)
  trainset = torchvision.datasets.SVHN(root='./data', train=True,
                                          download=True, transform=transform)


  train_data = read_csv(args.train_path, sep=',', header=None).values
  trainset = DatasetFM(train_data, transforms=transform)
  trainset_, valaset = torch.utils.data.random_split(trainset, dataset_split)
  train_dataloader = torch.utils.data.DataLoader(trainset_, batch_size=batch_size,
                                            shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(valaset, batch_size=batch_size,
                                            shuffle=True)


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
      print('===================================== Epoch %d =====================================' % epoch_item)
      train_loss = 0.
      for i, batch in enumerate(train_dataloader):
        images, labels = batch
        # imshow(images)
        
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
            for j, data in enumerate(val_dataloader):
              sample, labels = data
              sample, labels = sample.to(device), labels.to(device)

              logits, _ = model.forward(sample)
    
              loss = criterion(logits, labels)
              loss = loss.mean()
              total_val_loss += loss.item()

            total_val_loss /= len(val_dataloader)
            print('Epoch: %d/%d, Train Loss: %f, Val Loss: %f' % (
              epoch_item, i+1,  train_loss/args.log_interval, total_val_loss))
            print('===============================================')
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


def batch_test(model, args, device):
  
  # transform_test = transforms.Compose([
  #   transforms.ToPILImage(),
  #   transforms.ToTensor(),
  #   transforms.Normalize((0.5071, 0.4867, 0.4408),
  #                       (0.2675, 0.2565, 0.2761)),
  # ])
  
  # test_data = read_csv(args.test_path, sep=',', header=None).values
  # # test_dataset = DatasetFM(test_data, transforms=transform_test)
  # test_dataset = DatasetFM(test_data)
  # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
  #                                       download=True, transform=transform)
  test_data = read_csv(args.test_path, sep=',', header=None).values
  testset = DatasetFM(test_data, transforms=transform)
  test_dataloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                          shuffle=False)


  # == Load model & Detector
  if args.which_model == 'best':
    try:
      model.load(args.best_model_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.best_model_path))
  elif args.which_model == 'last':
    try:
      model.load(args.last_model_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.last_model_path))
  model.to(device)


  correct = 0
  total = 0

  model.eval()
  with torch.no_grad():
    for i, data in enumerate(test_dataloader):
  
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      out, feature = model.forward(sample)
      
      _, predicted = torch.max(out.data, 1)
      total += label.size(0)
      correct += (predicted == label).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %7.4f %%' % (100 * correct / total))

  

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


