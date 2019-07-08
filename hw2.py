# -*- coding:utf-8 -*-
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def read_data(self, fileName):
        name = fileName[fileName.rfind('/') + 1:]
        data = np.genfromtxt(fileName, delimiter=',', skip_header=1)
        if name == "X_train":
            self.x_train = data
        elif name == "Y_train":
            self.y_train = data
        elif name == "X_test":
            self.x_test = data

    @staticmethod
    def normalize(x, isTrain, special_col=None, mean=None, std=None):
        if isTrain:
            if special_col == None:
                special_col = np.arange(x.shape[1])
            length = len(special_col)
            X_mean = np.reshape(np.mean(x[:, special_col], 0), (1, length))
            X_std = np.reshape(np.std(x[:, special_col], 0), (1, length))

            x[:, special_col] = np.divide(
                np.subtract(x[:, special_col], X_mean), X_std)

            return X_mean, X_std

    @staticmethod
    def train_validate_split(x, y, percent=0.15):
        length = int(np.floor(x.shape[0] * (1 - percent)))
        x_train = x[0: length]
        y_train = y[0: length]
        x_dev = x[length:]
        y_dev = y[length:]
        return x_train, y_train, x_dev, y_dev

    @staticmethod
    def _sigmoid(z):
        val = np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return val

    @staticmethod
    def _get_prob(x, w, b):
        return Solution._sigmoid(x.dot(w) + b)

    @staticmethod
    def _infer(x, w, b):
        return np.round(Solution._get_prob(x, w, b))

    @staticmethod
    def _cross_entropy(y_pred, y_label):
        cross_entropy = -y_label.dot(np.log(y_pred)) - \
            (1-y_label).dot(np.log(1-y_pred))
        return cross_entropy

    @staticmethod
    def _loss(y_pred, y_label, lamda, w):
        return Solution._cross_entropy(y_pred, y_label) + np.sum(lamda * (w ** 2))

    @staticmethod
    def _accuracy(y_pred, y_label):
        return np.sum(y_pred == y_label) / len(y_pred)

    @staticmethod
    def train(x, y, lamda):
        x_train, y_train, x_dev, y_dev = Solution.train_validate_split(x, y)
        w = np.zeros(x_train.shape[1])
        b = np.zeros((1,))
        learn_rate = 0.2
        lr_w = np.zeros(len(w))
        lr_b = 0
        maxIter = 200

        train_acc = []
        train_loss = []
        dev_acc = []
        dev_loss = []

        for i in range(maxIter):
            y_pred = Solution._get_prob(x_train, w, b)
            w_grad = -x_train.T.dot(y_train - y_pred) + lamda * w
            b_grad = -np.sum(y_train - y_pred)
            lr_w += w_grad ** 2
            lr_b += b_grad ** 2
            w = w - learn_rate/np.sqrt(lr_w) * w_grad
            b = b - learn_rate/np.sqrt(lr_b) * b_grad

            y_train_pred = Solution._get_prob(x_train, w, b)
            Y_train_pred = np.round(y_train_pred)
            train_acc.append(Solution._accuracy(Y_train_pred, y_train))
            train_loss.append(Solution._loss(
                y_train_pred, y_train, lamda, w) / len(y_train))

            y_dev_pred = Solution._get_prob(x_dev, w, b)
            Y_dev_pred = np.round(y_dev_pred)
            dev_acc.append(Solution._accuracy(Y_dev_pred, y_dev))
            dev_loss.append(Solution._loss(
                y_dev_pred, y_dev, lamda, w) / len(y_dev))

        return w, b, train_acc, train_loss, dev_acc, dev_loss

    @staticmethod
    def plot_result(train_acc, train_loss, dev_acc, dev_loss):
        plt.plot(train_loss)
        plt.plot(dev_loss)
        plt.title('loss')
        plt.legend(['train', 'dev'])
        plt.show()
        plt.plot(train_acc)
        plt.plot(dev_acc)
        plt.title('acc')
        plt.legend(['train', 'dev'])
        plt.show()


def main():
    sol = Solution()
    x_train_filename = "D:/data/hw2/X_train"
    y_train_filename = "D:/data/hw2/Y_train"
    x_test_filename = "D:/data/hw2/X_test"

    sol.read_data(x_train_filename)
    sol.read_data(y_train_filename)
    sol.read_data(x_test_filename)

    col = [0, 1, 3, 4, 5, 7, 10, 12, 25, 26, 27, 28]
    mean, std = sol.normalize(sol.x_train, isTrain=True, special_col=col)
    x_train, y_train, x_dev, y_dev = sol.train_validate_split(
        sol.x_train, sol.y_train)
    w, b, train_acc, train_loss, dev_acc, dev_loss = sol.train(
        x_train, y_train, 0)
    # Solution.plot_result(train_acc, train_loss, dev_acc, dev_loss)
    Solution.normalize(sol.x_test, isTrain=False,
                       special_col=col, mean=mean, std=std)
    result = Solution._infer(sol.x_test, w, b)
    with open("D:/data/hw2/testOutput.csv", 'w') as csv_file:
        csv_file.write('id,label\n')
        for i, v in enumerate(result):
            csv_file.write("%d,%d\n", (i + 1), v)


if __name__ == "__main__":
    main()
