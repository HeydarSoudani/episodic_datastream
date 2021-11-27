import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    self.dce = DCELoss(device, gamma=args.temp_scale)
    self.ce = torch.nn.CrossEntropyLoss()
    # self.proto = PrototypeLoss()

  def forward(self, features, outputs, labels, prototypes, n_query, n_classes):
    dce_loss = self.dce(features, labels, prototypes, n_query, n_classes)
    cls_loss = self.ce(outputs, labels.long())
    # pt_loss = self.proto(features, labels, prototypes)

    return self.lambda_1 * dce_loss +\
           self.lambda_2 * cls_loss
    # return self.lambda_1 * dce_loss +\
    #        self.lambda_2 * cls_loss +\
    #        self.lambda_3 * pt_loss



## Inc. ================================
class DCELoss_inc(nn.Module):
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
      # torch.arange(0, n_classes, device=self.device, dtype=torch.long)
      unique_labels
      .view(n_classes, 1, 1)
      .expand(n_classes, n_query, 1)
    )

    loss_val = -log_p_y.gather(2, target_inds).mean()
    return loss_val


class TotalLoss_inc(nn.Module):
  def __init__(self, device, args):
    super().__init__()
    self.args = args
    self.lambda_1 = args.lambda_1
    self.lambda_2 = args.lambda_2
    
    self.dce = DCELoss_inc(device, gamma=args.temp_scale)
    self.ce = nn.CrossEntropyLoss()

  def forward(self, features, outputs, labels, prototypes, n_query, n_classes):
    dce_loss = self.dce(features, labels, prototypes, n_query, n_classes)
    # cls_loss = self.ce(outputs, labels.long())
    cls_loss = self.ce(outputs, labels)

    print(cls_loss)

    return self.lambda_1 * dce_loss +\
           self.lambda_2 * cls_loss



# [[ 0.3376,  0.1666,  0.4220, -0.2586, -0.4544, -0.1168, -0.0924, -0.1531, -0.1256,  0.0411],[ 0.1952, -0.1048,  0.1441,  0.2189,  0.2905,  0.1015, -0.1399,  0.1302, -0.1257, -0.4267],[ 0.1458,  0.0853, -0.2644, -0.1576,  0.3733, -0.4543,  0.2643,  0.2575, -0.0775, -0.2474],[ 0.0910, -0.2113,  0.0527,  0.0694,  0.1918,  0.0861, -0.3315,  0.3087, -0.0867, -0.4285],[-0.3097,  0.1893,  0.1407, -0.0733,  0.0801, -0.2575, -0.1383, -0.0491, -0.1611,  0.1225], [ 0.8789, -0.3240, -0.1081,  0.5345, -0.1371, -0.0021, -0.4708,  0.2020, -0.1247, -0.6271], [ 0.8085,  0.3487,  0.8757,  0.7693,  0.1428, -0.2935, -0.8466, -0.1159, 0.0895, -0.8519], [ 0.7342,  0.0839,  0.3560, -0.1353,  0.1620, -0.5205, -0.4481, -0.4177, -0.2518, -0.1628], [ 0.9350, -0.0174,  0.2055,  0.3977,  0.1364, -0.4179, -0.6404,  0.0733, 0.0872, -0.1912], [ 0.4602, -0.1357,  0.4494,  0.0367,  0.4061, -0.0883, -0.1529, -0.3360, -0.1723, -0.2766]]










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
