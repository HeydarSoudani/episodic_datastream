import torch
from learners.pt_learner import compute_prototypes


## According to the article, this learner does not use query set
class ReptileLearner:
  def __init__(self, criterion, device, args):
    self.criterion = criterion
    self.device = device

    if args.dataset == 'mnist' \
      or args.dataset == 'fmnist'\
        or args.dataset == 'cifar10':
      class_num = 10
    elif args.dataset == 'cifar100':
      class_num = 100
    else:
      class_num = 0

    self.prototypes = {
      l: torch.zeros(1, args.hidden_dims, device=device)
      for l in range(class_num)
    }

  def train(self, model, queue, optimizer, iteration, args):
    model.train()
    old_vars = [param.data.clone() for param in model.parameters()]

    queue_length = len(queue)
    losses = 0

    for k in range(args.update_step):
      for i in range(queue_length):

          optimizer.zero_grad()

          batch = queue[i]
          support_images = batch['data']
          support_labels = batch['label']
          support_images = support_images.reshape(-1, *support_images.shape[2:])
          support_labels = support_labels.flatten() 
          support_images = support_images.to(self.device)
          support_labels = support_labels.to(self.device)
         
          logits, support_features = model.forward(support_images)

          # Update pts ========
          unique_label = torch.unique(support_labels)
          episode_prototypes = compute_prototypes(
            support_features, support_labels
          )
          old_prototypes = torch.cat(
            [self.prototypes[l.item()] for l in unique_label]
          )

          if args.beta_type == 'evolving':
            beta = args.beta * iteration / args.meta_iteration
          elif args.beta_type == 'fixed':
            beta = args.beta

          new_prototypes = beta * old_prototypes + (1 - beta) * episode_prototypes
          # ===================
          
          loss = self.criterion(logits, support_labels)
          loss.backward()
          losses += loss.detach().item()

          torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
          optimizer.step()

          # Update pts ========
          for idx, l in enumerate(unique_label):
            self.prototypes[l.item()] = new_prototypes[idx].reshape(1, -1).detach()
          # ===================

    beta = args.beta * (1 - iteration / args.meta_iteration)
    for idx, param in enumerate(model.parameters()):
      param.data = (1 - beta) * old_vars[idx].data + beta * param.data

    return losses / (queue_length * args.update_step)


  def evaluate(self, model, dataloader):
    ce = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
      total_loss = 0.0
      for i, batch in enumerate(dataloader):

        sample, labels = batch
        sample, labels = sample.to(self.device), labels.to(self.device)
        
        logits, features = model.forward(sample)
        # loss = criterion(features, logits, labels, prototypes)
        loss = ce(logits, labels)
        # loss, acc = criterion(features, target=labels)
        loss = loss.mean()
        total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss
