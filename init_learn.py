
from trainers import train
# from detectors.reptile_detector import replite_detector
from detectors.pt_detector import pt_detector
from utils.memory_selector import OperationalMemory

from datasets.dataset import DatasetFM
from torch.utils.data import DataLoader
from utils.preparation import transforms_preparation

def init_learn(model,
               memory,
               train_data,
               base_labels,
               args, device):

  ### == Train Model =================
  train(model, train_data, args, device)

  ## == Save Novel detector ==========
  if args.which_model == 'best':
    model.load(args.best_model_path)
  # _ = replite_detector(model, train_data, args, device)
  _ = pt_detector(model, train_data, base_labels, args, device)


  ## == Save Memory selector ==========
  _, test_transform = transforms_preparation()
  if args.use_transform:
    dataset = DatasetFM(train_data, transforms=test_transform)
  else:
    dataset = DatasetFM(train_data)
  dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

  features = []
  for i, data in enumerate(dataloader):
    sample, label = data
    sample, label = sample.to(device), label.to(device)
    _, feature = model.forward(sample)
    features.append((feature.detach(), label.item()))

  memory.select(data=features)
  memory.save(args.memory_path)
  print("Memory has been saved in {}.".format(args.memory_path))

if __name__ == '__main__':
  pass