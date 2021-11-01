import torch


## According to the article, this learner does not use query set
def reptile_learner(model,
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    criterion, criterion_mt,
                    optimizer,
                    args):
  #Support_set: [shot_num*ways, C, H, W], [shot_num*ways]
  #Query_set:   [q_num*ways,    C, H, W], [q_num*ways]
  model.train()

  old_vars = []
  running_vars = []
  for param in model.parameters():
    old_vars.append(param.data.clone())
  
  losses = 0.

  for k in range(args.update_step):
    optimizer.zero_grad()
    logits, _ = model.forward(support_images)
    loss_cls = criterion(logits, support_labels)
    loss_mt = criterion_mt(logits, support_labels)

    loss_cls = loss_cls.mean() 
    loss_mt = loss_mt.mean()

    loss = 0.5 * loss_cls + loss_mt
    loss.backward()
    losses += loss
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
  
  if running_vars == []:
    for _,param in enumerate(model.parameters()):
      running_vars.append(param.data.clone())
  else:
    for idx,param in enumerate(model.parameters()):
      running_vars[idx].data += param.data.clone()

  for idx,param in enumerate(model.parameters()):
    param.data = old_vars[idx].data.clone()

  for idx,param in enumerate(model.parameters()):
    param.data = old_vars[idx].data + args.beta * (running_vars[idx].data - old_vars[idx].data)

  return losses/args.update_step


def reptile_evaluate(model, dataloader, criterion, device):
  with torch.no_grad():
    total_loss = 0.0
    model.eval()
    for i, data in enumerate(dataloader):

      sample, labels = data
      sample, labels = sample.to(device), labels.to(device)
      
      logits, _ = model.forward(sample)
      
      loss = criterion(logits, labels)
      loss = loss.mean()
      total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss