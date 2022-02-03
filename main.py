import os
import torch
import argparse
import numpy as np
from pandas import read_csv

from datasets.dataset import SimpleDataset
from models.cnn import Conv_4
from models.mlp import MLP
from models.resnet import ResNet18, Resnet50

from utils.memory_selector import OperationalMemory, IncrementalMemory
from detectors.pt_detector import PtDetector
from learners.pt_learner import PtLearner
from learners.reptile_learner import ReptileLearner
from learners.batch_learner import BatchLearner
from losses import TotalLoss, MetricLoss

from phases.init_learn import init_learn
from phases.zeroshot_test import zeroshot_test
from phases.stream_learn import stream_learn
# from phases.incremental_learn import batch_increm_learn, episodic_increm_learn

from plot.class_distribution import class_distribution
from plot.feature_space_visualization import set_novel_label, visualization


## == Params ===========
parser = argparse.ArgumentParser()

parser.add_argument(
  '--phase',
  type=str,
  choices=[
    'init_learn',
    'zeroshot_test',
    'stream_learn',
    'zeroshot_test_base',
    'batch_incremental_learn',
    'episodic_incremental_learn',
    'plot'
  ],
  default='plot',
  help='')
parser.add_argument(
  '--which_model',
  type=str,
  choices=['best', 'last'],
  default='best',
  help='')
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'pmnist',
    'rmnist',
    'fmnist',
    'pfmnist',
    'rfmnist',
    'cifar10',
    'cifar100'
  ],
  default='cifar100',
  help='') 
parser.add_argument(
  '--algorithm',
  type=str,
  choices=['prototype', 'reptile', 'batch'],
  default='prototype',
  help='')

# init train
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--meta_iteration', type=int, default=3000, help='')
parser.add_argument('--log_interval', type=int, default=100, help='must be less then meta_iteration parameter')
parser.add_argument('--ways', type=int, default=5, help='')
parser.add_argument('--shot', type=int, default=5, help='')
parser.add_argument('--query_num', type=int, default=5, help='')
parser.add_argument('--buffer_size', type=int, default=1000, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')

# retrain
parser.add_argument('--retrain_epochs', type=int, default=1, help='')
parser.add_argument('--retrain_meta_iteration', type=int, default=1000, help='')
parser.add_argument('--known_retrain_interval', type=int, default=5000, help='')
parser.add_argument('--known_per_class', type=int, default=100, help='for known buffer')

# incremental learning
parser.add_argument('--n_tasks', type=int, default=5, help='')

# Network
parser.add_argument('--dropout', type=float, default=0.2, help='')
# parser.add_argument('--hidden_dims', type=int, default=128, help='')

# memory
parser.add_argument('--mem_sel_type', type=str, default='fixed_mem', choices=['fixed_mem', 'pre_class'], help='')
parser.add_argument('--mem_total_size', type=int, default=2000, help='')
parser.add_argument('--mem_per_class', type=int, default=100, help='')
parser.add_argument('--mem_sel_method', type=str, default='rand', choices=['rand', 'soft_rand'], help='')
parser.add_argument('--mem_novel_acceptance', type=int, default=150, help='')

# Transform
parser.add_argument('--use_transform', action='store_true')

# Prototypical algorithm
parser.add_argument('--beta_type', type=str, choices=['fixed', 'evolving'], default='evolving', help='Update Prototype in Prototypical algorithm')
parser.add_argument('--beta', type=float, default=1.0, help='Update Prototype in Prototypical algorithm')
parser.add_argument('--drift_beta', type=float, default=1.0, help='')
parser.add_argument('--std_coefficient', type=float, default=1.0, help='for Pt detector')

# Reptile algorithm
parser.add_argument('--update_step', type=int, default=5, help='for Reptile algorithm')

# Loss function
parser.add_argument("--lambda_1", type=float, default=1.0, help="DCE Coefficien in loss function")
parser.add_argument("--lambda_2", type=float, default=1.0, help="CE Coefficient in loss function")
parser.add_argument("--lambda_3", type=float, default=0.0001, help="Metric Coefficient in loss function")
parser.add_argument("--temp_scale", type=float, default=0.2, help="Temperature scale for DCE in loss function",)

# Optimizer
parser.add_argument('--lr', type=float, default=0.1, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=1e-4, help='')  #l2 regularization
parser.add_argument('--grad_clip', type=float, default=0.1)   # before was 5.0

# Scheduler
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument("--step_size", default=8, type=int)
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
parser.add_argument('--learner_path', type=str, default='saved/learner.pt', help='')

# WideResNet Model
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.2, type=float, help='aspect of erasing area')

# visualization
parser.add_argument('--vis_filename', default='tsne', type=str, help='feature_space_plotting')

args = parser.parse_args()

## == Add some variables to args =======
args.data_path = 'data/'
args.train_file = '{}_train.csv'.format(args.dataset)
args.test_file = '{}_test.csv'.format(args.dataset)
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
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)

## == Set class number =================
if args.dataset in ['mnist', 'pmnist', 'rmnist', 'fmnist', 'pfmnist', 'rfmnist', 'cifar10']:
  args.n_classes = 10
  args.hidden_dims = 128
elif args.dataset in ['cifar100']:
  args.n_classes = 100
  args.hidden_dims = 160

## == Save dir =========================
if not os.path.exists(args.save):
  os.makedirs(args.save)

## == Model Definition =================
if args.dataset in ['mnist', 'pmnist', 'rmnist']:
  # MLP net selected like CoPE
  model = MLP(784, args.hidden_dims, args.n_classes, args)
elif args.dataset == 'cifar100':
  # model = ResNet18(100, args)
  model = Resnet50(args)
else:
  model = Conv_4(args)

## == Load model if exist ==============
if args.phase not in ['init_learn']:
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

## == Loss & Learner Definition =========
if args.algorithm == 'prototype':
  criterion = TotalLoss(device, args)
  learner = PtLearner(criterion, device, args)
elif args.algorithm == 'reptile':
  criterion = torch.nn.CrossEntropyLoss()
  learner = ReptileLearner(criterion, device, args)
elif args.algorithm == 'batch':
  criterion = MetricLoss(device, args)
  learner = BatchLearner(criterion, device, args)

try: learner.load(args.learner_path)
except FileNotFoundError: pass
else: print("Load Learner from {}".format(args.learner_path))

# == For stream version ==================
if args.phase not in [
  'batch_incremental_learn',
  'episodic_incremental_learn']:

  ## == load train data from file ========
  train_data = read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',',
    header=None).values
  base_labels = SimpleDataset(train_data, args).label_set

  ## == Operational Memory Definition ====
  memory = OperationalMemory(device,
                            selection_type=args.mem_sel_type,
                            total_size=args.mem_total_size,
                            per_class=args.mem_per_class,
                            novel_acceptance=args.mem_novel_acceptance)
  try: memory.load(args.memory_path)
  except FileNotFoundError: pass
  else: print("Load Memory from {}".format(args.memory_path))

  ## == Novelty Detector Definition =======
  detector = PtDetector()
  try: detector.load(args.detector_path)
  except FileNotFoundError: pass
  else: print("Load Detector from {}".format(args.detector_path))

if __name__ == '__main__':
  ## == Data Stream ====================
  if args.phase == 'init_learn':
    init_learn(model,
               learner,
               memory,
               detector,
               train_data,
               args, device)
  elif args.phase == 'zeroshot_test':
    zeroshot_test(model,
                  learner.prototypes,
                  detector,
                  args, device)
  elif args.phase == 'stream_learn':
    stream_learn(model,
                 learner,
                 memory,
                 detector,
                 args, device)
  elif args.phase == 'zeroshot_test_base':
    zeroshot_test(model,
                  learner.prototypes,
                  detector,
                  args, device,
                  known_labels=base_labels)
  
  # ## == incremental learning ============
  # elif args.phase == 'batch_incremental_learn':
  #   memory = IncrementalMemory(
  #             selection_type=args.mem_sel_type, 
  #             total_size=args.mem_total_size,
  #             per_class=args.mem_per_class,
  #             selection_method=args.mem_sel_method)
  #   batch_increm_learn(model,
  #                     memory,
  #                     args,
  #                     device)
  # elif args.phase == 'episodic_incremental_learn':
  #   memory = IncrementalMemory(
  #             selection_type=args.mem_sel_type, 
  #             total_size=args.mem_total_size,
  #             per_class=args.mem_per_class,
  #             selection_method=args.mem_sel_method)
  #   episodic_increm_learn(model,
  #                       learner,
  #                       memory,
  #                       args,
  #                       device)
  
  ## == Plot ===========================
  elif args.phase == 'plot':
    
    # = Class distribution in stream dataset
    # class_distribution(args)
    
    # # = data in feature-space after training
    stream_data_novel = set_novel_label(base_labels, args)
    visualization(model, stream_data_novel, args, device, filename=args.vis_filename)

  else: 
    raise NotImplementedError()








