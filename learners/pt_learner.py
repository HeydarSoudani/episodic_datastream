import torch
import time

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
  def __init__(self, criterion, device):
    self.criterion = criterion
    self.device = device
    self.prototypes = None

  def train(self, model, batch, optimizer, iteration, args):
    model.train()  
    optimizer.zero_grad()

    support_len = args.shot * args.ways

    support_images, support_labels, query_images, query_labels = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    query_images = query_images.reshape(-1, *query_images.shape[2:])
    query_labels = query_labels.flatten()
    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    images = torch.cat((support_images, query_images))
    outputs, features = model.forward(images)
    
    new_prototypes = compute_prototypes(
      features[:support_len], support_labels
    )

    beta = args.beta * iteration / args.meta_iteration
    if iteration > 1 and beta > 0.0:
      self.prototypes = beta * self.prototypes + (1 - beta) * new_prototypes
    else:
      self.prototypes = new_prototypes

    loss = self.criterion(
      features[support_len:],
      outputs[support_len:],
      query_labels,
      self.prototypes,
      n_query=args.query_num,
      n_classes=args.ways,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    if beta > 0.0:
      self.prototypes = self.prototypes.detach()
    else:
      self.prototypes = None

    return loss.detach().item()

  #TODO: classification with distance metric
  def evaluate(self, model, dataloader):
    
    ce = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
      total_loss = 0.0
      model.eval()
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
