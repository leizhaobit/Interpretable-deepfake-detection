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
import torchvision


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
    correct = 0
    for step, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        with torch.no_grad():
            logits = net(inputs)
            logits = torch.sigmoid(logits)
            logits[logits >= 0.5] = 1
            logits[logits < 0.5] = 0

            correct += torch.all(torch.eq(logits, labels.unsqueeze(-1)), dim=1).sum()
            #print("step: ", step)
            #print(correct / ((step+1)*int(inputs.shape[0])))
            #accuracy = torch.all(torch.eq(logits, labels),  dim=1).sum()/len(labels)
            #print(accuracy)
        if step % 20 == 0:
            #print("logits:", logits, "labels:", labels)
            log(
                f'\tAccuracy@top1 ({step}/{len(dataloader)}) = {correct / ((step + 1) * int(inputs.shape[0])):.5%}')
            # log(f'\tAccuracy@top3 ({step}/{len(dataloader)}) = {correct_top3/((step+1)*int(inputs.shape[0])):.5%}')
            # log(f'\tAccuracy@top5 ({step}/{len(dataloader)}) = {correct_top5/((step+1)*int(inputs.shape[0])):.5%}')

    return correct / ((step + 1) * int(inputs.shape[0]))


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
    net = torchvision.models.mobilenet_v2(num_classes=1).cuda()
    state_dict = torch.load(pretrained_model).state_dict()
    net.load_state_dict(state_dict)

    cudnn.benchmark = True
    accuracy = 0


    dataset = 'celeb-df2'
    testset = my_dataloader('/home/zhaolei2/project/xception/datasets/'+dataset, split='test')
    testloader = DataLoader(testset, batch_size=8, shuffle=True, collate_fn=testset.my_collate)

    temp_accuracy = eval(net, testloader)
    print('acc:', temp_accuracy)



if __name__ == "__main__":
    clean()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model='/home/zhaolei2/project/xception/paper/build/mobilenet_v2_ff.pt')
