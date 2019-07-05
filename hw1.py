# -*- coding: utf-8 -*-

import numpy as np
import sys


def LoadData(fileName):
    rawData = np.genfromtxt(fileName, delimiter=',', encoding='utf-8')
    data = rawData[1:, 3:]
    whereNaNs = np.isnan(data)
    data[whereNaNs] = 0
    return data


def PreProcess(data):

    month_to_data = {}  # Dictionary (key:month , value:data)

    for month in range(12):
        sample = np.empty(shape=(18, 480))
        for day in range(20):
            for hour in range(24):
                sample[:, day * 24 + hour] = data[18 *
                                                  (month * 20 + day): 18 * (month * 20 + day + 1), hour]
        month_to_data[month] = sample

    x = np.empty(shape=(12 * 471, 18 * 9), dtype=float)
    y = np.empty(shape=(12 * 471, 1), dtype=float)

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_to_data[month][:,
                                                                           day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                y[month * 471 + day * 24 + hour,
                    0] = month_to_data[month][9, day * 24 + hour + 9]
    return x, y


def Normalization(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0:
                x[i][j] = (x[i][j] - mean[j]) / std[j]
    return mean, std


def TrainModel(x, y):
    dim = x.shape[1] + 1
    w = np.zeros(shape=(dim, 1))
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1).astype(float)
    learning_rate = np.array([[200]] * dim)
    adagrad_sum = np.zeros(shape=(dim, 1))

    for T in range(10000):
        if(T % 500 == 0):
            print("T=", T)
            print("Loss:", np.power(
                np.sum(np.power(x.dot(w) - y, 2)) / x.shape[0], 0.5))
        gradient = (-2) * np.transpose(x).dot(y-x.dot(w))
        adagrad_sum += gradient ** 2
        w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)
    return w


def Test(testFileName, weight, mean, std):
    testRawData = np.genfromtxt(testFileName, delimiter=',', encoding='utf-8')
    testData = testRawData[:, 2:]
    testData[np.isnan(testData)] = 0
    testX = np.zeros(shape=(240, 18 * 9), dtype=float)
    for i in range(240):
        testX[i] = testData[18 * i: 18 * (i + 1), :].reshape(1, -1)
    for i in range(testX.shape[0]):
        for j in range(testX.shape[1]):
            if not std[j] == 0:
                testX[i][j] = (testX[i][j] - mean[j]) / std[j]

    testX = np.concatenate(
        (testX, np.ones(shape=(testX.shape[0], 1))), axis=1).astype(float)
    res = testX.dot(weight)
    return res


#main
trainDataFileName = "D:/data/train.csv"
testDataFileName = "D:/data/test.csv"
data = LoadData(trainDataFileName)
x, y = PreProcess(data)
mean, std = Normalization(x)
w = TrainModel(x, y)
answer = Test(testDataFileName, w, mean, std)
print(answer)
