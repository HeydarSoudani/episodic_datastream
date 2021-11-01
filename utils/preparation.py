from torch.utils.data import DataLoader
import numpy as np

from augmentation import transforms
from datasets.dataset import DatasetFM
from sampler import DataSampler

# from torchvision.transforms.transforms import RandomRotation
# from augmentation.autoaugment.autoaugment import CIFAR10Policy

def dataloader_preparation(train_data, args):
  print(train_data.shape)

  n, _ = train_data.shape
  np.random.shuffle(train_data)
  train_val_data = np.split(train_data, [int(n*0.9), n])
  train_data = train_val_data[0]
  val_data = train_val_data[1]

  ## ==========================
  train_transform, test_transform = transforms_preparation()
  
  if args.use_transform:
    train_dataset = DatasetFM(train_data, transforms=train_transform)
    val_dataset = DatasetFM(val_data, transforms=test_transform)
  else:
    train_dataset = DatasetFM(train_data)
    val_dataset = DatasetFM(val_data)
    
  train_sampler = DataSampler(
    train_dataset,
    n_way=args.ways,
    n_shot=args.shot,
    n_query=args.query_num,
    n_tasks=args.meta_iteration
  )
  train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
  )

  val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
  
  return train_dataloader, val_dataloader


def transforms_preparation():

  train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4, fill=128),
    # transforms.RandomHorizontalFlip(p=0.5),
    # CIFAR10Policy(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])

  test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  return train_transform, test_transform


