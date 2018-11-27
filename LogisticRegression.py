# -*- coding=utf-8 -*-
from cmath import exp

import shape as shape


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()     #convert classLabels to matrix, then get it's 转置矩阵
    m, n = shape(dataMatIn)
    alpha = 0.001 # 学习步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatIn*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatIn.transpose() * error
    return weights





