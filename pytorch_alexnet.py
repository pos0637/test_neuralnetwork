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

# 定义网络结构


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),


])

transform1 = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=0)  # windows下num_workers设置为0，不然有bug

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    # net
    net = AlexNet(num_classes=10)

    # 损失函数:这里用交叉熵
    criterion = nn.CrossEntropyLoss()

    # 优化器 这里用SGD
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    net.to(device)
    net.train()

    print("Start Training!")

    num_epochs = 20  # 训练次数

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
    torch.save(net, 'MNIST.pkl')


# train()
net = torch.load('MNIST.pkl')
net.eval()

# 开始识别
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
        for image in images:
            out = net(image.unsqueeze(0).to(device))
            _, predicted = torch.max(out.data, 1)
            print(predicted)

            img = image.clone().mul(255).byte().numpy().transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('image', img)
            cv2.waitKey(0)
