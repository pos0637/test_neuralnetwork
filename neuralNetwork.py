# -*- coding: UTF-8 -*-

import numpy as np
import scipy


class NeuralNetwork:
    """神经网络
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """构造函数

        Args:
            input_nodes (Number): 输入层节点数量
            hidden_nodes (Number): 隐藏层节点数量
            output_nodes (Number): 输出层节点数量
            learning_rate (Number): 学习率
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.wih = np.random.normal(
            0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(
            0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input, output):
        """训练

        Args:
            input (Array): 输入数据
            output (Array: 输出数据
        """
        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activation_function(hidden_input)
        output_input = np.dot(self.who, hidden_output)
        output_output = self.activation_function(output_input)


if __name__ == '__main__':
    nn = NeuralNetwork(3, 3, 3, 0.01)
