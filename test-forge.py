import imageio
import os
import shutil
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append('.')  # noqa: E402
from model import RACNN
from my_loader_test import my_dataloader
from pretrain_apn import random_sample, save_img, clean, log, build_gif



def avg(x): return sum(x)/len(x)


def train(net, dataloader, optimizer, epoch, _type):
    assert _type in ['apn', 'backbone']
    losses = 0
    net.mode(_type), log(f' :: Switch to {_type}')  # switch loss type
    for step, (inputs, targets) in enumerate(dataloader, 0):
        loss = net.echo(inputs, targets, optimizer)
        losses += loss

        if step % 20 == 0 and step != 0:
            avg_loss = losses/20
            log(f':: loss @step({step:2d}/{len(dataloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
            losses = 0

    return avg_loss


def eval(net, dataloader):
    log(':: Testing on test set...')
    correct = 0.0
    correct_summary = {'clsf-0': 0, 'clsf-1': 0,
                       'clsf-2': 0}
    acc_summary = {'clsf-0': 0, 'clsf-1': 0,
                       'clsf-2': 0}
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            for idx, logits in enumerate(outputs):
                logits = torch.sigmoid(logits)
                logits[logits >= 0.5] = 1
                logits[logits < 0.5] = 0
                correct_summary[f'clsf-{idx}'] += torch.all(torch.eq(logits, labels.unsqueeze(-1)), dim=1).sum()
        if step % 200 == 0 and step != 0:
            #print("logits:", logits, "labels:", labels)
            for clsf in correct_summary.keys():
                acc_summary[clsf] = correct_summary[clsf] / ((step + 1) * int(inputs.shape[0]))
                log(
                    f'\tAccuracy {clsf}@ ({step}/{len(dataloader)}) = {correct_summary[clsf] / ((step + 1) * int(inputs.shape[0])):.5%}')
    for clsf in correct_summary.keys():
        correct += acc_summary[clsf]
    acc = correct / 3
    return acc


def test(net, dataloader):
    log(' :: Testing on test set ...')
    correct_summary = {'clsf-0': {'top-1': 0, 'top-5': 0}, 'clsf-1': {'top-1': 0, 'top-5': 0}, 'clsf-2': {'top-1': 0, 'top-5': 0}}
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            outputs, _, _, _ = net(inputs)
            for idx, logits in enumerate(outputs):
                correct_summary[f'clsf-{idx}']['top-1'] += torch.eq(logits.topk(max((1, 1)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-1
                correct_summary[f'clsf-{idx}']['top-5'] += torch.eq(logits.topk(max((1, 5)), 1, True, True)[1], labels.view(-1, 1)).sum().float().item()  # top-5

            if step > 200:
                for clsf in correct_summary.keys():
                    _summary = correct_summary[clsf]
                    for topk in _summary.keys():
                        log(f'\tAccuracy {clsf}@{topk} ({step}/{len(dataloader)}) = {_summary[topk]/((step+1)*int(inputs.shape[0])):.5%}')
                return


def run(pretrained_model):
    log(f' :: Loading with {pretrained_model}')
    net = RACNN(num_classes=1).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    cudnn.benchmark = True
    accuracy = 0


    dataset = 'DeeperForensics'
    testset = my_dataloader('/home/zhaolei2/project/xception/datasets/'+dataset, split='test')
    testloader = DataLoader(testset, batch_size=8, shuffle=True, collate_fn=testset.my_collate)

    temp_accuracy = eval(net, testloader)
    print('acc:', temp_accuracy)



if __name__ == "__main__":
    clean()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model=f'build/racnn_ffc23.pt')
