# coding=utf-8
import numpy as np
import struct
import os
import time

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input # 输入维度 
        self.num_output = num_output # 输出维度
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    def forward(self, input):  # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果 
        self.output = self.input.dot(self.weight) + self.bias 
        return self.output 
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        '''
            self.d_weight = np.matmul(self.input.T, top_diff) # dL/dW
            self.d_bias = np.matmul(np.ones([self.input.shape[0], 1]).T, top_diff) # dL/db
            bottom_diff = np.matmul(top_diff, self.weight.T) # dL/dX
        '''
        self.d_weight = np.matmul(self.input.T, top_diff) # dL/dW
        self.d_bias = np.matmul(np.ones([self.input.shape[0], 1]).T, top_diff) # dL/db
        bottom_diff = np.matmul(top_diff, self.weight.T) # dL/dX
        return bottom_diff
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        self.weight = self.weight - lr * self.d_weight # W = W - lr * dL/dW
        self.bias = self.bias - lr * self.d_bias # b = b - lr * dL/db
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self):  # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.') 
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        # relu = max(0, input)
        output = np.maximum(0, self.input) # ReLU  
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        bottom_diff = np.where(self.input > 0, top_diff, 0) # ReLU
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True) # softmax
        return self.prob
    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        # softmax = (prob - label_onehot) / batch_size
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff



