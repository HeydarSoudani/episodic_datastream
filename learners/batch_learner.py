import torch

class BatchLearner:
  def __init__(self, criterion, device, args):
    self.criterion = criterion
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.hidden_dims, device=device)
      for l in range(args.n_classes)
    }

  
  def train(self, model, batch, optimizer, args):
    model.train()  
    optimizer.zero_grad()

    images, labels = batch
    images, labels = images.to(self.device), labels.to(self.device)

    ## == Forward ===========================
    outputs, _ = model.forward(images)
    loss = self.criterion(outputs, labels)
    
    ## == Calculate Prototypes ==============


    ## == Backward ==========================
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    return loss.detach().item()
  
  def evaluate(self, model, dataloader, args):
    model.eval()
    
    total_loss = 0.0
    total_dist_acc = 0.0
    correct_cls_acc = 0.0
    total_cls_acc = 0

    with torch.no_grad():
      for j, data in enumerate(dataloader):
        sample, labels = data
        sample, labels = sample.to(device), labels.to(device)

        logits, _ = model.forward(sample)

        ## == Cls-based Acc. ===================
        _, predicted = torch.max(logits, 1)
        total_cls_acc += labels.size(0)
        correct_cls_acc += (predicted == labels).sum().item()
        
        ## == loss =============================
        loss = criterion(logits, labels)
        loss = loss.mean()
        total_loss += loss.item()
      
      total_loss /= len(dataloader)
      # total_dist_acc /= len(dataloader)
      total_cls_acc = correct_cls_acc / total_cls_acc  

    return total_loss, total_cls_acc

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)
