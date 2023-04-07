import os
from torch import nn, optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
from model_cpu import RACNN
#from plant_loader import get_plant_loader
from pretrain_apn import clean, log
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from CUB_loader import CUB200_loader
from my_loader import my_dataloader
import warnings
warnings.filterwarnings('ignore')
num_classes = 1

inp1 = torch.randn([8,2])
print(inp1.shape)
fla = nn.Flatten()
dense = nn.Linear(num_classes*2, num_classes)
acvi = nn.Sigmoid()
x = fla(inp1)
x = dense(x)
out1 = acvi(x)
print(out1.shape)