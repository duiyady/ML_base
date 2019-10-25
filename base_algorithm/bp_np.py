# -*- coding:utf-8 -*-
# @Time: 2019-10-25 10:00
# @Author: duiya duiyady@163.com


import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, input_num, learn_rate):
        self.W = []
        self.b = []
        self.a = []
        self.input_num = input_num
        self.last_num = None
        self.x = None
        self.y = None
        self.pre = None
        self.learn_rate = learn_rate
        self.length = 0

    def add_layer(self, num):
        if self.last_num is None:
            tmp_w = -1 + 2*np.random.random((num, self.input_num))
            tmp_b = -1 + 2*np.random.random((num, 1))
        else:
            tmp_w = -1 + 2*np.random.random((num, self.last_num))
            tmp_b = -1 + 2 * np.random.random((num, 1))
        self.W.append(tmp_w)
        self.b.append(tmp_b)
        self.last_num = num
        self.length += 1

    def get_loss(self):
        return np.mean(0.5 * pow(self.pre - self.y, 2))

    def forward(self):
        assert self.x.shape[1] == self.input_num
        a_tmp = self.x
        for w, b in zip(self.W, self.b):
            self.a.append(a_tmp)
            a_tmp = (np.dot(w, a_tmp.T) + b).T
        self.pre = a_tmp

    def backford(self):
        tmp_td = []
        tmp = None
        i = self.length - 1
        while i >= 0:
            if tmp is None:
                tmp = self.pre - self.y
            else:
                tmp = np.dot(tmp, self.W[i+1])
            tmp_td.append(tmp)
            i -= 1
        for i in range(self.length):
            now_gradent = tmp_td[self.length - i - 1]
            self.W[i] -= self.learn_rate*np.dot(now_gradent.T, self.a[i])/now_gradent.shape[0]
            self.b[i] -= self.learn_rate*np.mean(now_gradent, axis=0).reshape(self.b[i].shape)

    def predict(self, x):
        self.x = x
        self.forward()
        return self.pre

    def summary(self):
        print('==================================')
        print('总层数：', self.length, '\t输入：', self.input_num)
        for i in range(len(self.W)):
            print('第', i+1, '层:', self.W[i].shape[0])
        print('==================================')

    def train(self, x, y):
        self.x = x
        self.y = y
        self.forward()
        print('loss:', self.get_loss())
        self.backford()


def create_samble():
    x_data = np.linspace(-2, 2, 1000)[:, np.newaxis]
    np.random.shuffle(x_data)
    noise = np.random.normal(-0.1, 0.1, x_data.shape)
    y_data = 0.3 * x_data + 0.8 + noise
    return x_data, y_data


if __name__ == '__main__':
    model = Model(input_num=1, learn_rate=0.01)
    model.add_layer(10)
    model.add_layer(10)
    model.add_layer(1)
    model.summary()

    epochs = 100

    x, y = create_samble()
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    for i in range(epochs):
        model.train(x, y)
    plt.scatter(x, y)
    pre = model.predict(x)
    plt.scatter(x, pre, color='red')
    plt.show()











