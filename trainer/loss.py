import logging

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR

from typing import Any, Callable

import torch
import numpy as np 
import torch.nn as nn 

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

import torch.nn.functional as F
from functools import partial


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)

class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present'):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes

    def forward(self, probas, labels):
        return lovasz_softmax_flat(probas, labels[0], self.classes)

class CrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1, weight=None):
        super(MixLovaszCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight = weight)
    
    def forward(self, x, y, n):
        ce = self.ce(x, y)
        return ce

class MixLovaszCrossEntropy(nn.Module):
    def __init__(self, classes='present', ignore_index=-1, weight=None,arch="KPConv"):
        super(MixLovaszCrossEntropy, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.lovasz = Lovasz_softmax(classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.arch = arch
    
    def forward(self, x, y, n):
        if not self.arch == "SPVCNN":
            x_lovasz = x[0].T
            lovasz = self.lovasz(F.softmax(x_lovasz, 1), y)
        else:
            lovasz = lovasz_softmax_flat(F.softmax(x, 1), y, self.lovasz.classes)
        ce = self.ce(x, y)
        return lovasz + ce

