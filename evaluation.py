import argparse

import numpy as np
import os
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score


def get_parser():
  """"Defines the command line arguments"""
  parser = argparse.ArgumentParser(description='Open World Vision')
  parser.add_argument('--pred_file', required=True, type=str,
                      help='path to csv file containing predicted labels. First column should contain image name and'
                            'rest of the columns should contain predicted probabilities for each of the class_id')
  parser.add_argument('--gt_file', required=True, type=str,
                      help='path to csv file containing ground-truth labels. First column should contain image name'
                            ' and second column should contain the ground truth class_id')
  parser.add_argument('--k_vals', default=1, nargs='+', type=int,
                      help='space separated list of k(s) in top-k evaluation')
  return parser


def roc(scores, labels, **kwargs):
  """Returns the ROC curve and the area under it. The implementation assumes binary classification
  Args:
    labels (np.array): denotes label for each data point
    scores (np.array): denotes predicted probability of data point being 1
  """
  if 'gt_shift' in kwargs:
    labels = labels - kwargs['gt_shift']
  fpr, tpr, thresholds = roc_curve(labels, scores)
  auc_score = roc_auc_score(labels, scores)
  roc_data = {
    'tp': list(tpr),
    'fp': list(fpr),
    'thresh': list(thresholds),
    'auc': auc_score,
  }
  return roc_data


def pr(scores, labels, **kwargs):
  """Returns the PR curve and the area under it. The implementation assumes binary classification
  Args:
    labels (np.array): denotes label for each data point
    scores (np.array): denotes predicted probability of data point being 1
  """
  precision, recall, thresholds = precision_recall_curve(labels, scores)
  auc_score = auc(recall, precision)
  pr_data = {
    'precision': list(precision),
    'recall': list(recall),
    'thresh': list(thresholds),
    'auc': auc_score,
  }
  return pr_data


# Function based on the definition given in the paper: https://arxiv.org/pdf/1811.04110.pdf
# Adapted from https://github.com/Vastlab/Reducing-Network-Agnostophobia/blob/master/Tools/evaluation_tools.py
def ccr(scores, labels, **kwargs):
  """Returns the CCR VS FPR curve and returns the area under it.
  Args:
    labels (np.array): denotes label for each data point
    scores (np.array): denotes predicted probability of data point being 1
  """
  if 'gt_shift' in kwargs:
    labels = labels - kwargs['gt_shift']
  max_indices = np.argmax(scores, axis=1)
  max_values = np.take_along_axis(scores, max_indices[:, None], axis=1)
  data = list(zip([elem[0] for elem in max_values], max_indices, labels))
  data.sort(key=lambda x: x[0], reverse=True)
  cls_count = len(scores[0])
  unknown_count = len(np.where(labels == 0)[0])  # Unknown class is 0
  known_count = len(labels) - unknown_count
  fp = [0]
  tp = [0]
  thresholds = list()
  n = 0
  n_above_unknown = 0
  curr_unknown_prob = 1
  for score, pred_cls, gt_cls in data:
    if gt_cls == 0:
      curr_unknown_prob = score
      thresholds.append(score)
      fp.append(fp[-1] + 1)
      tp.append(n)
    elif pred_cls == gt_cls:
      n_above_unknown += 1
      if score < curr_unknown_prob:
        n = n_above_unknown
  fpr = np.asarray(fp[1:]) / unknown_count
  ccr = np.asarray(tp[1:]) / known_count
  auc_score = auc(fpr, ccr)
  ccr_data = {
    'ccr': list(ccr),
    'fpr': list(fpr),
    'thresh': thresholds,
    'auc': auc_score,
  }
  return ccr_data


def macro_F1(scores, labels, binary=False, **kwargs):
    """Calculates the f1 score
    Args:
        labels (np.array): denotes label for each data point
        scores (np.array): denotes predicted probability of data point being 1
        binary (bool): flag denoting if labels are binary or multi class
    """
    preds = np.argsort(-scores)[:, :1]
    if 'gt_shift' in kwargs:
        labels = labels - kwargs['gt_shift']
    if binary:
        # f1 score for openset evaluation
        tp = 0.
        fp = 0.
        fn = 0.
        for i in range(len(labels)):
            tp += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            fp += 1 if preds[i] != labels[i] and labels[i] != -1 else 0
            fn += 1 if preds[i] != labels[i] and labels[i] == -1 else 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:  # Regular f1 score
        f1score = f1_score(labels, preds, average='macro')

    return f1score


def check_class_validity(p_class, gt_class):
    """
    Check the validity of the inputs for image classification. Raises an exception if invalid.
    Args:
        p_class (np.array): Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
        gt_class (np.array) : Nx1 vector with ground-truth class for each sample
    """
    if p_class.shape[0] != gt_class.shape[0]:
        raise Exception(
            'Number of predicted samples not equal to number of ground-truth samples!')
    if np.any(p_class < 0) or np.any(p_class > 1):
        raise Exception(
            'Predicted class probabilities should be between 0 and 1!')
    if p_class.ndim != 2:
        raise Exception(
            'Predicted probabilities must be a 2D matrix but is an array of dimension {}!'.format(p_class.ndim))
    if np.max(gt_class) >= p_class.shape[1] or np.min(gt_class < 0):
        raise Exception(
            'Ground-truth class labels must lie in the range [0-{}]!'.format(p_class.shape[1]))
    return


def top_k_accuracy(p_class, gt_class, k, gt_shift=0, mode='all'):
  """
  The method computes top-K accuracy.
  Args:
    p_class: Nx(K+1) matrix with each row corresponding to K+1 class probabilities for each sample
    gt_class: Nx1 compute vector with ground-truth class for each sample
    k (int): 'k' used in top-K accuracy
    gt_shift (int): amount by which gt_labels should be shifted down before comparing with predictions. This is
                    because predictions are always indexed from 0
    mode (str): 'all' - computes combined accuracy on all classes
                else computes individual class accuracies
  Returns:
    top-K accuracy
  """

  def calc_accuracy(pred, gt, k):
    pred = np.argsort(-pred)[:, :k]
    gt = gt[:, np.newaxis]
    check_zero = pred - gt
    correct = np.sum(np.any(check_zero == 0, axis=1).astype(int))
    return round(float(correct) / pred.shape[0], 5)

  gt_class = gt_class - gt_shift
  check_class_validity(p_class, gt_class)
  accuracy = dict()
  if mode == 'all':
    accuracy['all'] = calc_accuracy(p_class, gt_class, k)
  else:
    class_list = sorted(map(int, np.unique(gt_class)))
    for class_id in class_list:
      class_inds = np.where(gt_class == class_id)
      class_gt = gt_class[class_inds]
      class_p = p_class[class_inds]
      accuracy[class_id + gt_shift] = calc_accuracy(class_p, class_gt, k)
  return accuracy


def get_accuracy(preds, labels, k=(1, 5,), **kwargs):
  accuracy_list = dict()
  for k_val in k:
    class_accuracies = top_k_accuracy(preds, labels, k_val, mode='class', gt_shift=kwargs['gt_shift'])
    class_accuracies.update(top_k_accuracy(preds, labels, k_val, gt_shift=kwargs['gt_shift']))
    accuracy_list[k_val] = class_accuracies
  return accuracy_list


def evaluate(results, known_labels, k=(1, 5,), eps=1e-8):
  results = np.array(results, dtype=[
    ('true_label', np.int32),
    ('predicted_label', np.int32),
    ('real_novelty', np.bool),
    ('detected_novelty', np.bool)
  ])
  print('in eval new_labels: {}'.format(known_labels))
  # == Unknown Detection ==========
  real_novelties = results[results['real_novelty']]
  detected_novelties = results[results['detected_novelty']]
  detected_real_novelties = results[results['detected_novelty'] & results['real_novelty']]
  tp = len(detected_real_novelties)
  fp = len(detected_novelties) - len(detected_real_novelties)
  fn = len(real_novelties) - len(detected_real_novelties)
  tn = len(results) - tp - fp - fn
  M_new = fn / (tp + fn + eps)
  F_new = fp / (fp + tn + eps)

  # == Close-World ================
  known_results = results[np.isin(results['true_label'], list(known_labels))]
  cm = confusion_matrix(
    known_results['true_label'],
    known_results['predicted_label'],
    sorted(list(np.unique(results['true_label'])))
  )
  CwCA = accuracy_score(
    known_results['true_label'],
    known_results['predicted_label']
  )

  # == Open-World =================
  unknown_results = results[np.isin(results['true_label'], list(known_labels), invert=True)]
  unknown_results['true_label'] = -1
  ow_results = np.concatenate((known_results, unknown_results))
  OwCA = accuracy_score(
    ow_results['true_label'],
    ow_results['predicted_label']
  )


  return M_new, F_new, CwCA, OwCA, cm



if __name__ == '__main__':
  pass