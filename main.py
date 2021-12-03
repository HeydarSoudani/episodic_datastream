import os
import time
import torch
import argparse
import numpy as np
from pandas import read_csv

from datasets.dataset import SimpleDataset
from models.cnn import Conv_4
from models.densenet import DenseNet
from models.wrn import WideResNet

from trainers.train_batch import batch_train, batch_test
from utils.memory_selector import OperationalMemory, IncrementalMemory
from detectors.pt_detector import PtDetector
from learners.pt_learner import PtLearner
from losses import TotalLoss
# from pytorch_metric_learning import distances, losses, miners
from init_learn import init_learn
from zeroshot_test import zeroshot_test
from stream_learn import stream_learn
from incremental_learn import increm_learn
from utils.plot_tsne import plot_tsne



## == Params ===========
parser = argparse.ArgumentParser()

parser.add_argument('--phase', type=str, default='init_learn', help='')
parser.add_argument('--which_model', type=str, default='best', help='')
parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10]

# init train
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--meta_iteration', type=int, default=3000, help='')
parser.add_argument('--log_interval', type=int, default=100, help='must be less then meta_iteration parameter')
parser.add_argument('--ways', type=int, default=5, help='')
parser.add_argument('--shot', type=int, default=5, help='')
parser.add_argument('--query_num', type=int, default=5, help='')
parser.add_argument('--buffer_size', type=int, default=1000, help='')

# retrain
parser.add_argument('--retrain_epochs', type=int, default=1, help='')
parser.add_argument('--retrain_meta_iteration', type=int, default=1000, help='')
parser.add_argument('--known_retrain_interval', type=int, default=5000, help='')
parser.add_argument('--known_per_class', type=int, default=100, help='')

# incremental learning
parser.add_argument('--n_tasks', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--inc_memory', type=int, default=2000, help='')

# Network
parser.add_argument('--dropout', type=float, default=0.2, help='')
parser.add_argument('--hidden_dims', type=int, default=128, help='') #768

# memory
parser.add_argument('--memory_per_class', type=int, default=250, help='')
parser.add_argument('--memory_novel_acceptance', type=int, default=150, help='')

# Transform
parser.add_argument('--use_transform', action='store_true')

# Prototypical algorithm
parser.add_argument('--beta', type=float, default=1.0, help='Update Prototype in Prototypical algorithm')
parser.add_argument('--std_coefficient', type=float, default=1.0, help='for Pt detector')

# Reptile algorithm
parser.add_argument('--update_step', type=int, default=5, help='for Reptile algorithm')

# Loss function
parser.add_argument("--lambda_1", type=float, default=1.0, help="DCE Coefficient in loss function")
parser.add_argument("--lambda_2", type=float, default=1.0, help="CE Coefficient in loss function")
parser.add_argument("--lambda_3", type=float, default=0.001, help="PT Coefficient in loss function")
parser.add_argument("--temp_scale", type=float, default=0.2, help="Temperature scale for DCE in loss function",)

# Optimizer
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=0.0005, help='')  #l2 regularization
parser.add_argument('--grad_clip', type=float, default=5.0)

# Scheduler
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument("--step_size", default=3000, type=int)
parser.add_argument('--gamma', type=float, default=0.5, help='for lr step')

# Device and Randomness
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--seed', type=int, default=2, help='')

# Save and load model
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--best_model_path', type=str, default='saved/model_best.pt', help='')
parser.add_argument('--last_model_path', type=str, default='saved/model_last.pt', help='')
parser.add_argument('--best_mclassifier_path', type=str, default='saved/mclassifier_best.pt', help='for l2ac')
parser.add_argument('--last_mclassifier_path', type=str, default='saved/mclassifier_last.pt', help='for l2ac')

# Utils path
parser.add_argument('--memory_path', type=str, default='saved/memory.pt', help='')
parser.add_argument('--detector_path', type=str, default='saved/detector.pt', help='')
parser.add_argument('--prototypes_path', type=str, default='saved/prototypes.pt', help='')

# WideResNet Model
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.2, type=float, help='aspect of erasing area')

args = parser.parse_args()

## == Add some variables to args
args.data_path = 'data/{}'.format(args.dataset)
args.train_file = '{}_train.csv'.format(args.dataset)
args.stream_file = '{}_stream.csv'.format(args.dataset)
args.split_train_path = 'data/split_{}/train'.format(args.dataset)
args.split_test_path = 'data/split_{}/test'.format(args.dataset)

## == Device ===========================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Device: {}'.format(device))

## == Apply seed =======================
torch.manual_seed(args.seed)
np.random.seed(args.seed)

## == Save dir =========================
if not os.path.exists(args.save):
  os.makedirs(args.save)

## == Model Definition =================
model = Conv_4(args)
# model = DenseNet(args, tensor_view=(3, 32, 32))

# TODO: add init. weight
# model.apply(weights_init)

## == Load model if exist ==============
if args.which_model == 'best':
  try: model.load(args.best_model_path)
  except FileNotFoundError: pass
  else:
    print("Load model from {}".format(args.best_model_path))
elif args.which_model == 'last':
  try: model.load(args.last_model_path)
  except FileNotFoundError: pass
  else:
    print("Load model from {}".format(args.last_model_path))
model.to(device)


## == Loss & Learner Definition ========
# criterion  = nn.CrossEntropyLoss()
# criterion_mt = losses.NTXentLoss(temperature=0.07)
# criterion = PrototypicalLoss(n_support=args.shot)
criterion = TotalLoss(device, args)

pt_learner = PtLearner(criterion, device, args)
try: pt_learner.load(args.prototypes_path)
except FileNotFoundError: pass
else: print("Load Prototypes from {}".format(args.prototypes_path))


if args.phase != 'incremental_learn':

  ## == load train data from file ========
  train_data = read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',',
    header=None).values
  base_labels = SimpleDataset(train_data, args).label_set


  ## == Operational Memory Definition ====
  memory = OperationalMemory(per_class=args.memory_per_class,
                            novel_acceptance=args.memory_novel_acceptance,
                            device=device)
  try: memory.load(args.memory_path)
  except FileNotFoundError: pass
  else: print("Load Memory from {}".format(args.memory_path))


  ## == Novelty Detector Definition =======
  detector = PtDetector()
  try: detector.load(args.detector_path)
  except FileNotFoundError: pass
  else: print("Load Detector from {}".format(args.detector_path))


if __name__ == '__main__':

  ## == Non-episodic ======================
  # batch_train(model, train_data, args, device)
  # batch_test(model, args, device)

  ## == Data Stream =======================
  if args.phase == 'init_learn':
    init_learn(model,
               pt_learner,
               memory,
               detector,
               train_data,
               args,
               device)
  elif args.phase == 'zeroshot_test':
    zeroshot_test(model,
                  pt_learner.prototypes,
                  detector,
                  args,
                  device)
  elif args.phase == 'stream_learn':
    stream_learn(model,
                 pt_learner,
                 memory,
                 detector,
                 args,
                 device)
  elif args.phase == 'zeroshot_test_base':
    zeroshot_test(model,
                  pt_learner.prototypes,
                  detector,
                  args,
                  device,
                  known_labels=base_labels)
  ## == incremental learning ============
  elif args.phase == 'incremental_learn':
    memory = IncrementalMemory(args.inc_memory, device)
    increm_learn(model,
                 pt_learner,
                 memory,
                 args,
                 device)
  else: 
    raise NotImplementedError()

  ## == Data Visualization ==============
  # set_novel_label(args)
  # plot_tsne(args, device)



