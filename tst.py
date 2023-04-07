import shutil
import cv2
import imageio
import os
import numpy as np
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
#import seaborn as sns
import matplotlib.pyplot as plt
import random
sys.path.append('.')  # noqa: E402
from model_cpu import RACNN
from my_loader import my_dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
def process(labels):
    return

def rank_loss(logits, targets, margin=0.05):
    preds = [x for x in logits]  # preds length equal to 3
    criterion1 = torch.nn.MarginRankingLoss(margin=0.05, reduction='sum')
    criterion2 = torch.nn.MarginRankingLoss(margin=0.05, reduction='sum')
    y = targets

    print(preds[0].shape)
    print(y.shape)
    result1 = criterion1(preds[0], preds[1], y)
    return criterion1(preds[0], preds[1], y)

input1 = torch.tensor([0.2, 0.8, 0.7])
print(input1)
input2 = torch.tensor([0.1, 0.94, 0.95])
print(input2)

logits = torch.stack([input1, input2])
logits = logits.unsqueeze(-1)
target = torch.tensor([0,1,0]).unsqueeze(-1)
target[target == 1] = -1
target[target == 0] = 1
print(rank_loss(logits, target))

input1 = torch.tensor([0.2, 0.8, 0.7])
print(input1)
input2 = torch.tensor([0.1, 0.94, 0.95])
print(input2)

target = torch.tensor([0, 1, 0])
target[target == 1] = -1
target[target == 0] = 1
loss = torch.nn.MarginRankingLoss(margin=0.05, reduction='sum')
out = loss(input1, input2, target)
print(out)