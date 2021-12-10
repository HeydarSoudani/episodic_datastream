from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import os
import time

from utils.preparation import dataloader_preparation


def train(model,
          learner,
          train_data,
          args, device,
          val_data=[]):
  model.to(device)

  train_dataloader,\
  val_dataloader, \
  known_labels =  dataloader_preparation(train_data, val_data, args)
  
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
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch %d ===' % epoch_item)
      train_loss = 0.
      trainloader = iter(train_dataloader)

      for miteration_item in range(args.meta_iteration):
        batch = next(trainloader)
        loss = learner.train(model,batch,optim,miteration_item,args)
        train_loss += loss

        ## == validation ==============
        if (miteration_item + 1) % args.log_interval == 0:
          
          train_loss_total = train_loss / args.log_interval
          train_loss = 0.

          # evalute on val_dataset
          val_loss_total, \
          val_acc_dis_total, \
          val_acc_cls_total = learner.evaluate(model, val_dataloader, known_labels, args)  # For Pt.
          
          # print losses
          print('=== Time: %.2f, Step: %d, Train Loss: %f, Val Loss: %f' % (
            time.time()-global_time, miteration_item+1, train_loss_total, val_loss_total))
          # print('===============================================')
          global_time = time.time()
    
          # save best model
          if val_loss_total < min_loss:
            model.save(os.path.join(args.save, "model_best.pt"))
            min_loss = val_loss_total
            print("= ...New best model saved")
    
      if args.scheduler:
        scheduler.step()

  except KeyboardInterrupt:
    print('skipping training')  
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  print("= ...New last model saved")

  # save learner
  learner.save(os.path.join(args.save, "prototypes.pt"))
  print("= ..Pts saved")


if __name__ == '__main__':
  pass



