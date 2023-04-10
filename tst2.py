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


in_size = 6
x = torch.stack([torch.arange(0, in_size)] * in_size).t()
x = torch.stack([x] * 2)
MAX = torch.ones([2, 1, 1]) * 3
mx = (x >= MAX).float()

a = torch.ones([8])
print(a.shape)
a = a.unsqueeze(-1)
print(a.shape)
print('done')