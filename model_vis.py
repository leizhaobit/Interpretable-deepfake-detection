#import torchstat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import mobilenet
import numpy as np


class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        def h(_x): return 1 / (1 + torch.exp(-10 * _x.float()))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            # modify1: ori:tl = tl if tl > (in_size/3) else in_size/3
            tl = tl if tl > (in_size/3) else in_size/3
            tl = tl if tl < (in_size*2/3) else in_size*2/3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, w_off: w_end, h_off: h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.upsample(before_upsample, size=(224, 224), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor, locs)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor, locs = self.saved_variables[0], self.saved_variables[1], self.saved_variables[2]

        tmp_x, tmp_y, tmp_l = locs[:, 0], locs[:, 1], locs[:, 2]
        MAX = torch.ones([grad_output.size(0)]) * 224
        MAX = MAX.cuda()
        MIN = torch.zeros([grad_output.size(0)])
        MIN = MIN.cuda()

        x_max = torch.min(tmp_x + tmp_l, MAX)
        x_max = x_max.unsqueeze(-1)
        x_max = x_max.unsqueeze(-1)
        x_min = torch.max(tmp_x - tmp_l, MIN)
        x_min = x_min.unsqueeze(-1)
        x_min = x_min.unsqueeze(-1)

        y_max = torch.min(tmp_y + tmp_l, MAX)
        y_max = y_max.unsqueeze(-1)
        y_max = y_max.unsqueeze(-1)
        y_min = torch.max(tmp_y - tmp_l, MIN)
        y_min = y_min.unsqueeze(-1)
        y_min = y_min.unsqueeze(-1)

        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        '''
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size/3*2)
        short_size = (in_size/3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size)+(x >= long_size)+(y < short_size)+(y >= long_size)) > 0).float()*2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))
        '''
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        x = torch.stack([x] * grad_output.size(0))
        y = torch.stack([y] * grad_output.size(0))

        x = x.cuda()
        y = y.cuda()

        mx_batch = (x >= x_max).float() - (x < x_min).float()
        my_batch = (y >= y_max).float() - (y < y_min).float()
        ml_batch = (((x < x_min) + (x >= x_max) + (y < y_min) + (y >= y_max)) > 0).float() * 2 - 1

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """

    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)


class RACNN(nn.Module):
    def __init__(self, num_classes, img_scale=448):
        super(RACNN, self).__init__()

        self.b1 = mobilenet.mobilenet_v2(num_classes=num_classes)
        self.b2 = mobilenet.mobilenet_v2(num_classes=num_classes)
        self.b3 = mobilenet.mobilenet_v2(num_classes=num_classes)
        self.classifier1 = nn.Linear(320, num_classes)
        self.classifier2 = nn.Linear(320, num_classes)
        self.classifier3 = nn.Linear(320, num_classes)
        self.feature_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.crop_resize = AttentionCropLayer()
        self.apn1 = nn.Sequential(
            nn.Linear(320 * 7 * 7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.apn2 = nn.Sequential(
            nn.Linear(320 * 7 * 7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.echo = None

    def forward(self, x):
        batch_size = x.shape[0]
        #rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False).cuda()
        #rescale_tl = torch.tensor([1, 1, 0.5], requires_grad=False).cuda()
        # forward @scale-1
        feature_s1 = self.b1.features[:-1](x)  # torch.Size([1, 320, 14, 14])
        pool_s1 = self.feature_pool(feature_s1)
        _attention_s1 = self.apn1(feature_s1.view(-1, 320 * 7 * 7))
        #attention_s1 = _attention_s1*rescale_tl
        attention_s1 = _attention_s1
        resized_s1 = self.crop_resize(x, attention_s1 * x.shape[-1])
        # forward @scale-2
        feature_s2 = self.b2.features[:-1](resized_s1)  # torch.Size([1, 320, 7, 7])
        pool_s2 = self.feature_pool(feature_s2)
        _attention_s2 = self.apn2(feature_s2.view(-1, 320 * 7 * 7))
        #attention_s2 = _attention_s2*rescale_tl
        attention_s2 = _attention_s2
        resized_s2 = self.crop_resize(resized_s1, attention_s2 * resized_s1.shape[-1])
        # forward @scale-3
        feature_s3 = self.b3.features[:-1](resized_s2)
        pool_s3 = self.feature_pool(feature_s3)
        pred1 = self.classifier1(pool_s1.view(-1, 320))     #torch.Szie([16,8])
        pred2 = self.classifier2(pool_s2.view(-1, 320))
        pred3 = self.classifier3(pool_s3.view(-1, 320))

        return [pred1, pred2, pred3], [feature_s1, feature_s2], [attention_s1, attention_s2], [resized_s1, resized_s2]

    def __get_weak_loc(self, features):
        ret = []   # search regions with the highest response value in conv5
        for i in range(len(features)):
            resize = 224
            response_map_batch = F.interpolate(features[i], size=[resize, resize], mode="bilinear").mean(1)  # mean alone channels
            ret_batch = []
            for response_map in response_map_batch:
                argmax_idx = response_map.argmax()
                ty = (argmax_idx % resize)
                argmax_idx = (argmax_idx - ty)/resize
                tx = (argmax_idx % resize)
                ret_batch.append([(tx*1.0/resize).clamp(min=0.25, max=0.75), (ty*1.0/resize).clamp(min=0.25, max=0.75), 0.25])  # tl = 0.25, fixed
            ret.append(torch.Tensor(ret_batch))
        return ret

    def __echo_pretrain_apn(self, inputs, optimizer):
        inputs = Variable(inputs).cuda()
        _, features, attens, _ = self.forward(inputs)
        weak_loc = self.__get_weak_loc(features)
        optimizer.zero_grad()
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0].cuda())
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1].cuda())
        loss = weak_loss1 + weak_loss2
        loss.backward()
        optimizer.step()
        return loss.item()

    '''
    @staticmethod
    def multitask_loss(logits, targets):
        loss = []
        for i in range(len(logits)):
            loss.append(F.binary_cross_entropy_with_logits(logits[i].squeeze(-1), targets.float()))
            #loss.append(F.cross_entropy(logits[i], targets))
        loss = torch.sum(torch.stack(loss))
        return loss
    '''
    @staticmethod
    def multitask_loss(logits, targets):
        loss = []
        for i in range(len(logits)):
            loss.append(F.binary_cross_entropy_with_logits(logits[i].squeeze(-1), targets.float()))
            # loss.append(F.cross_entropy(logits[i], targets))
        return loss

    @staticmethod
    def rank_loss(logits, targets, margin=0.05):
        preds = [torch.sigmoid(x) for x in logits]
        criterion1 = torch.nn.MarginRankingLoss(margin=margin, reduction='sum')
        criterion2 = torch.nn.MarginRankingLoss(margin=margin, reduction='sum')
        targets = targets.unsqueeze(-1)
        targets[targets == 1] = -1
        targets[targets == 0] = 1
        return criterion1(preds[0], preds[1], targets), criterion2(preds[1], preds[2], targets)

    def __echo_backbone(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optim1, optim2, optim3 = optimizer
        loss1, loss2, loss3 = self.multitask_loss(logits, targets)

        optim3.zero_grad()
        loss3.backward(retain_graph=True)
        optim3.step()

        optim2.zero_grad()
        loss2.backward(retain_graph=True)
        optim2.step()

        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        #optimizer.zero_grad()
        #loss = self.multitask_loss(logits, targets)
        #loss.backward()
        #optimizer.step()
        #return loss.item()
        return loss1.item() + loss2.item() + loss3.item()

    def __echo_apn(self, inputs, targets, optimizer):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        logits, _, _, _ = self.forward(inputs)
        optim1, optim2 = optimizer
        loss1, loss2 = self.rank_loss(logits, targets)

        optim2.zero_grad()
        loss2.backward(retain_graph=True)
        optim2.step()

        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        #optimizer.zero_grad()
        #loss = self.rank_loss(logits, targets)
        #loss.backward()
        #optimizer.step()
        #return loss.item()
        return loss1.item() + loss2.item()

    def mode(self, mode_type):
        assert mode_type in ['pretrain_apn', 'apn', 'backbone']
        if mode_type == 'pretrain_apn':
            self.echo = self.__echo_pretrain_apn
            self.eval()
        if mode_type == 'backbone':
            self.echo = self.__echo_backbone
            self.train()
        if mode_type == 'apn':
            self.echo = self.__echo_apn
            self.eval()


if __name__ == "__main__":
    net = RACNN(num_classes=1).cuda()
    #net = RACNN(num_classes=8)
    net.mode('pretrain_apn')
    optimizer = torch.optim.SGD(list(net.apn1.parameters()) + list(net.apn2.parameters()), lr=0.001, momentum=0.9)
    x2 = torch.randn(size=(16, 3, 224, 224))
    out = net(x2)
    for i in range(50):
        inputs = torch.rand(2, 3, 224, 224)
        print(f':: loss @step{i} : {net.echo(inputs, optimizer)}')
