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
from model_vis import RACNN
from my_loader import my_dataloader
from pretrain_apn import clean, log


data = ''
bs = 16
def avg(x): return sum(x)/len(x)


def random_sample(dataloader):
    for batch_idx, (inputs, target) in enumerate(dataloader, 0):
        #print(inputs.shape)
        #print(target)
        return inputs


def save_img(x, path, index=0,  annotation=''):
    fig = plt.gcf()  # generate outputs
    img = my_dataloader.tensor_to_img(x[index])
    img = img[:, :, ::-1]
    plt.imshow(img, aspect='equal'), plt.axis('off'), fig.set_size_inches(448/100.0/3.0, 448/100.0/3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator()), plt.gca().yaxis.set_major_locator(plt.NullLocator()), plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0), plt.margins(0, 0)
    plt.text(0, 0, annotation, color='white', size=4, ha="left", va="top", bbox=dict(boxstyle="square", ec='black', fc='black'))
    plt.savefig(path, dpi=300, pad_inches=0)    # visualize masked image


def build_gif(pattern='@2x', gif_name='pretrain_apn_cub200', cache_path='build/.cache'):
    # generate a gif, enjoy XD
    files = [x for x in os.listdir(cache_path) if pattern in x]
    files.sort(key=lambda x: int(x.split('@')[0].split('_')[-1]))
    gif_images = [imageio.imread(f'{cache_path}/{img_file}') for img_file in files]
    imageio.mimsave(f"build/{gif_name}{pattern}-{int(time.time())}.gif", gif_images, fps=8)


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
    log(f' :: Start training with {pretrained_model}')
    net = RACNN(num_classes=1).cuda()
    net.load_state_dict(torch.load(pretrained_model))
    cudnn.benchmark = True
    accuracy = 0
    #cls_params = list(net.b1.parameters()) + list(net.b2.parameters()) + list(net.b3.parameters()) + \
    #    list(net.classifier1.parameters()) + list(net.classifier2.parameters()) + list(net.classifier3.parameters())
    #apn_params = list(net.apn1.parameters()) + list(net.apn2.parameters())

    cls1_params = list(net.b1.parameters()) + \
                  list(net.classifier1.parameters())
    cls2_params = list(net.b2.parameters()) + \
                  list(net.classifier2.parameters())
    cls3_params = list(net.b3.parameters()) + \
                  list(net.classifier3.parameters())
    apn1_params = list(net.apn1.parameters())
    apn2_params = list(net.apn2.parameters())

    cls_opt = [
        optim.SGD(cls1_params, lr=0.001, momentum=0.9),
        optim.SGD(cls2_params, lr=0.001, momentum=0.9),
        optim.SGD(cls3_params, lr=0.001, momentum=0.9),
    ]
    apn_opt = [
        optim.SGD(apn1_params, lr=1e-6),
        optim.SGD(apn2_params, lr=1e-6),
    ]

    #cls_opt = optim.SGD(cls_params, lr=0.001, momentum=0.9)
    #cls_opt = optim.Adam(params=cls_params, lr=0.001)
    #apn_opt = optim.SGD(apn_params, lr=0.001, momentum=0.9)
    #apn_opt = optim.Adam(params=apn_params, lr=0.001)
    dataset = 'ff--c23'
    dataset2 = 'vis-forgery2'
    data = dataset
    trainset = my_dataloader('/home/zhaolei2/project/xception/datasets/'+dataset, split='train')
    testset = my_dataloader('/home/zhaolei2/project/xception/datasets/'+dataset, split='test')
    visset = my_dataloader('/home/zhaolei2/project/xception/datasets/'+dataset2, split='test')
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=trainset.my_collate)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, collate_fn=testset.my_collate)
    visloader = DataLoader(visset, batch_size=bs, shuffle=False, collate_fn=testset.my_collate)
    sample = random_sample(visloader)
    sample = Variable(sample).cuda()

    train_epoch = 80
    for epoch in range(train_epoch):
        print('epoch:', epoch)
        cls_loss = train(net, trainloader, cls_opt, epoch, 'backbone')
        rank_loss = train(net, trainloader, apn_opt, epoch, 'apn')
        #test(net, testloader)
        temp_accuracy = eval(net, testloader)
        print('temp_acc:',temp_accuracy)
        # visualize cropped inputs
        _, _, _, resized = net(sample)
        x1, x2 = resized[0].data, resized[1].data
        for i in range(sample.size(0)):
            save_img(x1, index=i, path=f'build/.cache/epoch_{epoch}@forge_{i}_2x.jpg', annotation=f'cls_loss = {cls_loss:.7f}, rank_loss = {rank_loss:.7f}')
            save_img(x2, index=i, path=f'build/.cache/epoch_{epoch}@forge_{i}_4x.jpg', annotation=f'cls_loss = {cls_loss:.7f}, rank_loss = {rank_loss:.7f}')

        if temp_accuracy > accuracy:
            print('old:',accuracy, 'new:', temp_accuracy)
            accuracy = temp_accuracy
            #torch.save(net.state_dict(), f'build/racnn_'+dataset+'-vis.pt')
        # save model per 10 epoches
        '''if epoch % 10 == 0 and epoch != 0:
            stamp = f'e{epoch}{int(time.time())}'
            torch.save(net.state_dict, f'build/racnn_mobilenetv2_cub200-e{epoch}s{stamp}.pt')
            log(f' :: Saved model dict as:\tbuild/racnn_mobilenetv2_cub200-e{epoch}s{stamp}.pt')
            torch.save(cls_opt.state_dict(), f'build/cls_optimizer-s{stamp}.pt')
            torch.save(apn_opt.state_dict(), f'build/apn_optimizer-s{stamp}.pt')'''


if __name__ == "__main__":
    clean()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(pretrained_model=f'build/racnn_pretrained_compare.pt')
    for i in range(bs):
        build_gif(pattern=f'@forge_{i}_2x', gif_name='racnn_'+data)
        build_gif(pattern=f'@forge_{i}_4x', gif_name='racnn_'+data)
