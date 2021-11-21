import torch

def compute_prototypes(
  support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
  """
  Compute class prototypes from support features and labels
  Args:
    support_features: for each instance in the support set, its feature vector
    support_labels: for each instance in the support set, its label
  Returns:
    for each label of the support set, the average feature vector of instances with this label
  """
  seen_labels = torch.unique(support_labels)

  # Prototype i is the mean of all instances of features corresponding to labels == i
  return torch.cat(
    [
      support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
      for l in seen_labels
    ]
  )


class PtLearner:
  def __init__(self, criterion, device, args):
    self.criterion = criterion
    self.device = device

    if args.dataset == 'mnist' \
      or args.dataset == 'fmnist'\
        or args.dataset == 'cifar10':
      class_num = 10
    else:
      class_num = 0

    self.prototypes = {
      l: torch.zeros(1, args.hidden_dims, device=device)
      for l in range(class_num)
    }

  def train(self, model, batch, optimizer, iteration, args):
    model.train()  
    optimizer.zero_grad()

    support_len = args.shot * args.ways

    support_images, support_labels, query_images, query_labels = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    query_images = query_images.reshape(-1, *query_images.shape[2:])
    query_labels = query_labels.flatten()

    unique_label = torch.unique(support_labels)

    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    images = torch.cat((support_images, query_images))
    outputs, features = model.forward(images)
    
    episode_prototypes = compute_prototypes(
      features[:support_len], support_labels
    )
    old_prototypes = torch.cat(
      [self.prototypes[l.item()] for l in unique_label]
    )

    beta = args.beta * iteration / args.meta_iteration
    new_prototypes = beta * old_prototypes + (1 - beta) * episode_prototypes

    loss = self.criterion(
      features[support_len:],
      outputs[support_len:],
      query_labels,
      new_prototypes,
      n_query=args.query_num,
      n_classes=args.ways,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    for idx, l in enumerate(unique_label):
      self.prototypes[l.item()] = new_prototypes[idx].reshape(1, -1).detach()
    
    return loss.detach().item()

  #TODO: classification with distance metric
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

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)
