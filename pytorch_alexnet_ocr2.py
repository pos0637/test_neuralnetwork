# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import cv2
from torchvision.transforms.transforms import Grayscale, Resize
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def preprocess(path):
    raw = Image.open(path)
    image = ImageOps.grayscale(raw)
    raw.close()

    (w, h) = image.size
    w = int(w * 227 / h)
    resized = image.resize((w, 227), Image.BILINEAR)
    image.close()

    output = Image.new('L', (227, 227), (255))
    output.paste(resized, (int((227 - w) / 2), 0))
    resized.close()

    return output


def show_samples(example_data, example_targets):
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# transform
transform = transforms.Compose([
    transforms.Grayscale(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomGrayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

transform1 = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# 加载数据
trainset = datasets.ImageFolder(
    root='./samples', loader=preprocess, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=16, shuffle=True, num_workers=0)

testset = datasets.ImageFolder(
    root='./output', loader=preprocess, transform=transform1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=16, shuffle=False, num_workers=0)

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    # net
    net = AlexNet(num_classes=36)

    # 损失函数:这里用交叉熵
    criterion = nn.CrossEntropyLoss()

    # 优化器 这里用SGD
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.to(device)
    net.train()

    print("Start Training!")

    num_epochs = 30  # 训练次数

    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('[%d, %5d] loss:%.4f' % (epoch+1, (i+1)*100, loss.item()))

    print("Finished Traning")

    # 保存训练模型
    torch.save(net, 'ocr.pkl')


examples = enumerate(trainloader)
batch_idx, (example_data, example_targets) = next(examples)
show_samples(example_data, example_targets)

examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)
show_samples(example_data, example_targets)

train()
net = torch.load('ocr.pkl')
net.eval()

# 开始识别
if False:
    with torch.no_grad():
        # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False
        correct = 0
        total = 0

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            out = net(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images:{}%'.format(
            100 * correct / total))  # 输出识别准确率

with torch.no_grad():
    for data in testloader:
        images, labels = data
        for i in range(0, labels.shape[0]):
            image = images[i]
            out = net(image.unsqueeze(0).to(device))
            _, predicted = torch.max(out.data, 1)

            if predicted.item() != labels[i].item():
                if predicted.item() < 10:
                    print(predicted.item())
                else:
                    print(chr(65 + predicted.item() - 10))

                img = image.clone().mul(255).byte().numpy().transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow('image', img)
                cv2.waitKey(0)
