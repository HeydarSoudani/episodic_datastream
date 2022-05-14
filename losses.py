import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners
import time

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)

def euclidean_dist(x, y):
  '''
  Compute euclidean distance between two tensors
  '''
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

class DCELoss(nn.Module):
  def __init__(self, device, gamma=0.05):
    super().__init__()
    self.gamma = gamma
    self.device = device

  def forward(self, features, labels, prototypes, n_query, n_classes):
    unique_labels = torch.unique(labels)
    features = torch.cat(
      [features[(labels == l).nonzero(as_tuple=True)[0]] for l in unique_labels]
    )

    dists = euclidean_dist(features, prototypes)
    # dists = (-self.gamma * dists).exp() 

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = (
      torch.arange(0, n_classes, device=self.device, dtype=torch.long)
      .view(n_classes, 1, 1)
      .expand(n_classes, n_query, 1)
    )

    loss_val = -log_p_y.gather(2, target_inds).mean()
    return loss_val

class TotalLoss(nn.Module):
  def __init__(self, device, args):
    super().__init__()
    self.args = args
    self.lambda_1 = args.lambda_1
    self.lambda_2 = args.lambda_2
    self.lambda_3 = args.lambda_3
    self.lambda_4 = args.lambda_4
    
    self.dce = DCELoss(device, gamma=args.temp_scale)
    self.ce = torch.nn.CrossEntropyLoss()
    self.metric = losses.NTXentLoss(temperature=0.07)
    # self.metric = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    # self.metric = losses.TripletMarginLoss(margin=0.05)

  def forward(self, features, outputs, labels, prototypes, n_query, n_classes):
    dce_loss = self.dce(features, labels, prototypes, n_query, n_classes)
    cls_loss = self.ce(outputs, labels.long())
    metric_loss = self.metric(outputs, labels.long())

    return self.lambda_1 * dce_loss +\
           self.lambda_2 * cls_loss +\
           self.lambda_3 * metric_loss +\
           self.lambda_4 * self.pl_regularization(features, prototypes, labels)
  
  def pl_regularization(self, features, prototypes, labels):
    print(features.shape)
    print(prototypes.shape)
    print(labels)
    distance=(features-torch.t(prototypes)[labels])
    distance=torch.sum(torch.pow(distance,2),1, keepdim=True)
    distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]
    return distance

class MetricLoss(nn.Module):
  def __init__(self, device, args):
    super().__init__()
    self.args = args
    self.lambda_1 = args.lambda_1 # Metric loss coef
    self.lambda_2 = args.lambda_2 # CE coef

    self.ce = torch.nn.CrossEntropyLoss()
    # self.miner = miners.BatchEasyHardMiner() # for ContrastiveLoss 
    # self.metric = losses.NTXentLoss(temperature=0.07)
    # self.metric = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    self.metric = losses.TripletMarginLoss(margin=0.05)
    
    
  def forward(self, logits, labels):
    cls_loss = self.ce(logits, labels.long())
    
    # loss with miner
    # miner_output = self.miner(logits, labels.long())
    # metric_loss = self.metric(logits, labels.long(), miner_output)

    # loss without minier
    metric_loss = self.metric(logits, labels.long())

  
    return self.lambda_1 * metric_loss +\
           self.lambda_2 * cls_loss






















#TODO: add this loss
## prototype loss (PL): "Robust Classification with Convolutional Prototype Learning"
class PrototypeLoss(nn.Module):
  def __init__(self):
    super().__init__()
    # self.weights = weights

  def forward(self, features, labels, prototypes):

    n = features.shape[0]

    seen_labels = torch.unique(labels)
    prototype_dic = {l.item(): prototypes[idx].reshape(1, -1) for idx, l in enumerate(seen_labels)}
    # print(prototype_dic)
    loss = 0.
    for idx, feature in enumerate(features):
      dists = euclidean_dist(feature.reshape(1, -1), prototype_dic[labels[idx].item()])      #[q_num, cls_num]
      loss += dists
    
    loss /= n
    return loss

class PairwiseLoss(nn.Module):
  def __init__(self, tao=1.0, b=1.0, beta=0.1):
    super().__init__()
    self.b = b
    self.tao = tao
    self.beta = beta

  def forward(self, features, labels, prototypes):
    q_num = features.shape[0]
    cls_num = prototypes.shape[0]

    dists = torch.cdist(features, prototypes)      #[q_num, cls_num]
    likes = torch.ones(q_num, cls_num).to('cuda:0')
    likes[torch.arange(q_num), labels] =  torch.ones(q_num).to('cuda:0')
    inputs = (self.b - likes*(self.tao - dists.pow(2))).flatten()
    
    pw_loss = torch.mean(torch.tensor([self._g(input) for input in inputs]))
    return pw_loss

  def _g(self, z):
    return (1 + (self.beta * z).exp()).log() / self.beta if z < 10.0 else z
    # return (1 + (self.beta * z).exp()).log() / self.beta