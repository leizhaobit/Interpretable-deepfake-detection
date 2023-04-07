import imageio
import os
import numpy as np
import sys
import torch
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
sys.path.append('.')  # noqa: E402
from my_loader import my_dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICE'] = "0"


def log(msg):
    open('build/core.log', 'a').write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]\t'+msg+'\n'), print(msg)


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


def run():
    state_dict = torchvision.models.mobilenet_v2(pretrained=False)
    state_dict.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'))
    state_dict = state_dict.state_dict()
    state_dict.pop('classifier.1.weight')
    state_dict.pop('classifier.1.bias')
    net = torchvision.models.mobilenet_v2(num_classes=1).cuda()

    state_dict['classifier.1.weight'] = net.state_dict()['classifier.1.weight']
    state_dict['classifier.1.bias'] = net.state_dict()['classifier.1.bias']
    net.load_state_dict(state_dict)
    cudnn.benchmark = True

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    trainset = my_dataloader('/home/zhaolei2/project/xception/datasets/ff--c23', split='train')
    testset = my_dataloader('/home/zhaolei2/project/xception/datasets/ff--c23', split='test')
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=trainset.my_collate)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.my_collate)

    log(' :: Start training ...')
    benchmark = 100
    avg_loss = 100
    for epoch in range(100):
        losses = 0
        for step, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs).cuda(), Variable(labels.float()).cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(-1), labels)
            loss.backward()
            optimizer.step()

            losses += loss
            if step % 20 == 0 and step != 0:
                avg_loss = losses/20
                log(f':: loss @step({step:2d}/{len(trainloader)})-epoch{epoch}: {loss:.10f}\tavg_loss_20: {avg_loss:.10f}')
                losses = 0
        acc = eval(net, testloader)
        log(f':: test accuracy: {acc} ::')
        if avg_loss < benchmark:
            benchmark = avg_loss
            # stamp = f'e{epoch}{int(time.time())}'
            log(f'avg_loss: {avg_loss}, save model as build/mobilenet_v2_ff.pt')
            torch.save(net, f'build/mobilenet_v2_ff.pt')
            torch.save(optimizer.state_dict, f'build/optimizer.pt')


if __name__ == '__main__':
    path='build'
    if not os.path.exists(path):
        os.makedirs(path)
    run()

