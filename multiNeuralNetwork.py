# -*- coding: UTF-8 -*-

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import ndim


class MultiNeuralNetwork:
    """多层神经网络
    """

    def __init__(self, input_layer_nodes, hidden_layer_nodes, output_layer_nodes, learning_rate):
        """构造函数

        Args:
            input_nodes (Number): 输入层节点数量
            hidden_nodes (Array): 隐藏层节点数量
            output_nodes (Number): 输出层节点数量
            learning_rate (Number): 学习率
        """
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes
        self.learning_rate = learning_rate
        self.wights = []
        self.activation_function = lambda x: scipy.special.expit(x)

        self.wights.append(np.random.normal(
            0.0, pow(self.hidden_layer_nodes[0], -0.5), (self.hidden_layer_nodes[0], self.input_layer_nodes)))
        for i in range(1, len(self.hidden_layer_nodes)):
            self.wights.append(np.random.normal(
                0.0, pow(self.hidden_layer_nodes[i], -0.5), (self.hidden_layer_nodes[i], self.hidden_layer_nodes[i - 1])))
        self.wights.append(np.random.normal(
            0.0, pow(self.output_layer_nodes, -0.5), (self.output_layer_nodes, self.hidden_layer_nodes[-1])))

    def train(self, input_data, output_data):
        """训练

        Args:
            input (Array): 输入数据
            output (Array: 输出数据
        """
        input_data = np.array(input_data, ndmin=2).T
        output_data = np.array(output_data, ndmin=2).T

        outputs = [input_data]
        next_input = np.dot(self.wights[0], input_data)
        outputs.append(self.activation_function(next_input))
        for i in range(1, len(self.wights)):
            next_input = np.dot(self.wights[i], outputs[-1])
            outputs.append(self.activation_function(next_input))

        error = output_data - outputs[-1]
        self.wights[-1] += self.learning_rate * np.dot(error * outputs[-1] * (
            1 - outputs[-1]), np.transpose(outputs[-2]))
        for i in range(len(self.wights) - 2, -1, -1):
            error = np.dot(self.wights[i + 1].T, error)
            self.wights[i] += self.learning_rate * np.dot(error * outputs[i + 1] * (
                1 - outputs[i + 1]), np.transpose(outputs[i]))

    def predict(self, input_data):
        """预测

        Args:
            input_data (Array): 输入数据

        Returns:
            Array: 预测结果
        """
        input_data = np.array(input_data, ndmin=2).T

        outputs = [input_data]
        next_input = np.dot(self.wights[0], input_data)
        outputs.append(self.activation_function(next_input))
        for i in range(1, len(self.wights)):
            next_input = np.dot(self.wights[i], outputs[-1])
            outputs.append(self.activation_function(next_input))

        return outputs[-1]


def train_data(filename, nn, epochs, dump=False):
    """训练

    Args:
        filename (String): 文件名
        nn (NeuralNetwork): 神经网络
        epochs (Number): 次数
        dump (bool, optional): 是否显示内容. Defaults to False.
    """
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    for _ in range(epochs):
        for row in lines:
            data = row.split(',')
            input_data = np.asfarray(data[1:]) / 255.0 * 0.99
            output_data = np.zeros(10) + 0.01
            output_data[int(data[0])] = 0.99
            if dump:
                image = np.reshape(np.asfarray(data[1:]), (28, 28))
                print(data[0])
                plt.imshow(image, cmap='Greys', interpolation='None')
                plt.show()
            nn.train(input_data, output_data)


def test_data(filename, nn, dump=True):
    """测试

    Args:
        filename (String): 文件名
        nn (NeuralNetwork): 神经网络
        dump (bool, optional): 是否显示内容. Defaults to True.
    """
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    for row in lines:
        data = row.split(',')
        input_data = np.asfarray(data[1:]) / 255.0 * 0.99
        if dump:
            image = np.reshape(np.asfarray(data[1:]), (28, 28))
            plt.imshow(image, cmap='Greys', interpolation='None')
            plt.show()
        result = nn.predict(input_data)
        print(f'input: {data[0]}, result: {np.where(result==np.max(result))}')


if __name__ == '__main__':
    nn = MultiNeuralNetwork(28 * 28, [200, 200, 200], 10, 0.1)
    train_data('./data_100.csv', nn, 50)
    test_data('./data_10.csv', nn)
