# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import ndim


class NeuralNetwork:
    """神经网络
    """

    def __init__(self, input_layer_nodes, hidden_layer_nodes, output_layer_nodes, learning_rate):
        """构造函数

        Args:
            input_nodes (Number): 输入层节点数量
            hidden_nodes (Number): 隐藏层节点数量
            output_nodes (Number): 输出层节点数量
            learning_rate (Number): 学习率
        """
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes
        self.learning_rate = learning_rate
        self.wih = np.random.normal(
            0.0, pow(self.hidden_layer_nodes, -0.5), (self.hidden_layer_nodes, self.input_layer_nodes))
        self.who = np.random.normal(
            0.0, pow(self.output_layer_nodes, -0.5), (self.output_layer_nodes, self.hidden_layer_nodes))
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))

    def train(self, input_data, output_data):
        """训练

        Args:
            input (Array): 输入数据
            output (Array: 输出数据
        """
        input_data = np.array(input_data, ndmin=2).T
        output_data = np.array(output_data, ndmin=2).T

        hidden_layer_input = np.dot(self.wih, input_data)
        hidden_layer_output = self.activation_function(hidden_layer_input)
        output_layer_input = np.dot(self.who, hidden_layer_output)
        output_layer_output = self.activation_function(output_layer_input)

        output_layer_error = output_data - output_layer_output
        self.who += self.learning_rate * np.dot(output_layer_error * output_layer_output * (
            1 - output_layer_output), hidden_layer_output.T)

        hidden_layer_error = np.dot(self.who.T, output_layer_error)
        self.wih += self.learning_rate * np.dot(hidden_layer_error * hidden_layer_output * (
            1 - hidden_layer_output), input_data.T)

    def predict(self, input_data):
        """预测

        Args:
            input_data (Array): 输入数据

        Returns:
            Array: 预测结果
        """
        input_data = np.array(input_data, ndmin=2).T

        hidden_layer_input = np.dot(self.wih, input_data)
        hidden_layer_output = self.activation_function(hidden_layer_input)
        output_layer_input = np.dot(self.who, hidden_layer_output)
        output_layer_output = self.activation_function(output_layer_input)

        return output_layer_output


def train_data(filename, nn, dump=False):
    """训练

    Args:
        filename (String): 文件名
        nn (NeuralNetwork): 神经网络
        dump (bool, optional): 是否显示内容. Defaults to False.
    """
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

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
    nn = NeuralNetwork(28 * 28, 10000, 10, 0.01)
    train_data('./data_100.csv', nn)
    test_data('./data_10.csv', nn)
