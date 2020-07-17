# -*- coding: UTF-8 -*-

import torch  # 导入pytorch
from torch import nn, optim  # 导入神经网络与优化器对应的类
import torch.nn.functional as F
from torchvision import datasets, transforms  # 导入数据集与数据预处理的方法
import matplotlib.pyplot as plt

# 数据预处理：标准化图像数据，使得灰度数据在-1到+1之间
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载Fashion-MNIST训练集数据，并构建训练集数据载入器trainloader,每次从训练集中载入64张图片，每次载入都打乱顺序
trainset = datasets.FashionMNIST(
    './dataset/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

# 下载Fashion-MNIST测试集数据，并构建测试集数据载入器trainloader,每次从测试集中载入64张图片，每次载入都打乱顺序
testset = datasets.FashionMNIST(
    './dataset/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层
        self.conv = nn.Sequential(
            # 第一层卷积
            # 如果使用mnist类的数据集，输入通道为1，如果使用的cifar等彩色图像的数据集，输入通道为3
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第二层卷积，开始减小卷积窗口
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层卷积
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # 第四层卷积
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # 第五层卷积
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc = nn.Sequential(
            # 第一层全连接层
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 第二层全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 第三层全连接层（输出层）
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        feature = self.conv(x)
        x = self.fc(feature.view(x.shape[0], -1))

        return x


def evaluate_accuracy(testloader, model):
    test_loss = 0
    accuracy = 0

    # 测试的时候不需要开自动求导和反向传播
    with torch.no_grad():
        # 关闭Dropout
        model.eval()

        for images, labels in testloader:
            images, labels = images.to(device), labels.cuda()
            results = model(images).argmax(dim=1)
            accuracy += (results == labels).float().sum().cpu().item()

        # 恢复Dropout
        model.train()

    return test_loss, accuracy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 对上面定义的Classifier类进行实例化
model = Classifier().to(device)

# 定义损失函数为负对数损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 优化方法为Adam梯度下降方法，学习率为0.003
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 对训练集的全部数据学习15遍，这个数字越大，训练时间越长
epochs = 15

# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses, test_losses = [], []

print('开始训练')
for e in range(epochs):
    running_loss = 0

    # 对训练集中的所有图片都过一遍
    for images, labels in trainloader:
        images, labels = images.to(device), labels.cuda()

        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()

        # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss, accuracy = evaluate_accuracy(testloader, model)
    print("训练集学习次数: {}/{}.. " . format(e + 1, epochs),
          "训练误差: {:.3f}.. " . format(running_loss / len(trainloader)),
          "测试误差: {:.3f}.. " . format(test_loss / len(testloader)),
          "模型分类准确率: {:.3f} ". format(accuracy / len(testloader)))

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend()
plt.show()
