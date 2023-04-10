import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random

class my_dataloader(data.Dataset):
    def __init__(self, root, split='train', seed=0, transform=None):
        std = 1. / 255.
        means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
        # 开始获取图像路径和对应标签
        self._imgpath = list()
        self._imglabel = list()
        self.idx2name = ['real', 'fake']
        if split == 'train':
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'validation')
        tmp = os.path.join(root, 'real')
        print(tmp)
        for subdir, dirs, files in os.walk(tmp):
            for file in files:
                self._imgpath.append(os.path.join(subdir, file))
                self._imglabel.append(0)
            break

        tmp = os.path.join(root, 'fake')
        for subdir, dirs, files in os.walk(tmp):
            for file in files:
                self._imgpath.append(os.path.join(subdir, file))
                self._imglabel.append(1)
            break
        random.seed(seed)
        random.shuffle(self._imgpath)
        random.seed(seed)
        random.shuffle(self._imglabel)

        # 调整图像大小，归一化
        if transform is None and split.lower() == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.RandomCrop([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=means,
                    std=[std]*3)
            ])
        elif transform is None and split.lower() == 'test':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=means,
                    std=[std]*3)
            ])
        else:
            print(" [*] Warning: transform is not None, Recomend to use default")
            pass

    def __getitem__(self, index):
        img = cv2.imread(self._imgpath[index])
        img = self.transform(img)
        cls = self._imglabel[index]
        return img, cls

    def __len__(self):
        return len(self._imgpath)

    @staticmethod
    def tensor_to_img(x, imtype=np.uint8):
        """"将tensor的数据类型转成numpy类型，并反归一化.

                Parameters:
                    input_image (tensor) --  输入的图像tensor数组
                    imtype (type)        --  转换后的numpy的数据类型
        """
        mean = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
        std = [1. / 255., 1. / 255., 1. / 255.]

        if not isinstance(x, np.ndarray):
            if isinstance(x, torch.Tensor):  # get the data from a variable
                image_tensor = x.data
            else:
                return x
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = x
        return image_numpy.astype(imtype)

    def idx_to_classname(self, idx):
        return self.idx2name[idx]

    def my_collate(self, batch):
        imgs = list()
        cls = list()
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls


if __name__ == '__main__':
    trainset = my_dataloader('G:\deepfakes\celeb-df\celeb-df\celeb-df2')
    trainloader = data.DataLoader(trainset, batch_size=32,
                                  shuffle=False, collate_fn=trainset.my_collate, num_workers=1)
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        print(cls)
        break

