import copy
import random
import time

import torch
import torch.nn as nn
import syft
import torch.optim as optim
from torch.autograd import Variable

import utils
from Funtions import *

import matplotlib.pyplot as plt


class Arguments:
    def __init__(self):
        self.batchSize = 64  # 训练批次大小
        self.testBatchSize = 1000
        self.epochs = 10
        self.lr = 0.01  # 学习率
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.modelType = "CNN"


args = Arguments()
# device = torch.device("cuda")
device = torch.device("cpu")
hook = syft.TorchHook(torch)


def init():
    """Initialize"""
    patternIdx = 1
    clientNum = 10  # 客户数量
    model = utils.MNIST_CNN_Net().to(device)
    trainLoader = torch.load("train_loader_" + str(patternIdx) +
                             "_" + str(args.batchSize) + ".pt")
    testLoader = torch.load("test_loader_" + str(patternIdx) +
                            "_" + str(args.batchSize) + ".pt")
    return patternIdx, clientNum, model, trainLoader, testLoader


def runFedSA(model, clientNum, client, secureClient, server, clientData, clientTarget, testLoader):
    model.train()                  # 设置模型为训练模式
    lossF = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
    M = 5                          # 文章中的M值
    prepareTime = getPrepareTime(clientNum)  # 模型准备时间##############################################################
    clientModel, clientOpt = {}, {}  # worker的优化器
    for i in range(1, clientNum + 1):
        clientModel[str(i)] = model.copy().send(client[str(i)])
        clientOpt[str(i)] = optim.SGD(params=clientModel[str(i)].parameters(), lr=args.lr)

    currentLostTime = copy.deepcopy(prepareTime)
    runtime = 0
    tau = {}  # 模型延迟
    tau_th = 999  # 模型延迟的阈值
    for i in range(1, clientNum + 1):
        tau[str(i)] = 0
    tik = time.time()
    accList = []
    for k in range(1, 500):
        print("\nEpoch: {:d}".format(k))
        serverModel = model.copy().send(server)
        chosenClients, currentLostTime, iterationTime = chooseClients(M, prepareTime, currentLostTime)      ###########
        runtime += iterationTime
        clientData, clientOpt, clientModel, clientTarget, lossF = \
            localUpdate(device, chosenClients, clientData, clientOpt, clientModel, clientTarget, lossF)
        # clientModel = localDP(device, clientModel, chosenClients)

        for i in chosenClients:  # 把模型移动到secure_worker做简单平均
            clientModel[str(i)].move(secureClient)
        serverModel.move(secureClient)
        model = globalAggregate(serverModel, chosenClients, clientNum, clientModel, tau, tau_th)
        acc = modelTest("1.1", device, args, model, runtime, clientNum, testLoader)
        accList.append(acc)
        # 分发模型####################################################################################################
        for choose_worker in chosenClients:
            clientModel[str(choose_worker)] = model.copy().send(client[str(choose_worker)])
            clientOpt[str(choose_worker)] = optim.SGD(params=clientModel[str(choose_worker)].parameters(),
                                                      lr=args.lr)
        # 更新延迟####################################################################################################
        for i in range(1, clientNum + 1):
            if str(i) in chosenClients:
                tau[str(i)] = 0
            else:
                tau[str(i)] += 1
        tok = time.time()
        print("Total running time: {:f}\n".format(tok - tik))

    plt.figure()
    plt.plot(range(len(accList)), accList)
    plt.ylabel('test accuracy')
    plt.savefig('1.1CNN_64_0.01_10_acc.png')


if __name__ == '__main__':
    patternIdx, clientNum, model, trainLoader, testLoader = init()
    client, secureClient, server, clientData, clientTarget = distributeData(hook, args, clientNum, trainLoader)
    runFedSA(model, clientNum, client, secureClient, server, clientData, clientTarget, testLoader)
