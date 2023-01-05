# 计算一些参数的值
import copy
import math
from math import *
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torch import nn
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM, MNIST_CNN_Net, CIFAR_CNN_Net
from models.Fed import FedAvg, FedSA
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.Functions import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, dataset=None, idxs=None, learning_rate=0.01):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=64, shuffle=True)
        self.idxs = idxs
        self.learning_rate = learning_rate

    def train(self, net):
        net.train()
        pre_net = copy.deepcopy(net)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.learning_rate)
        epoch_loss = []
        batch_loss = []
        tmp_grad, tmp_w = [], []
        glob_w, glob_grad = [], []    # 在pre上只训练一个数据，几乎无差别
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
            log_probs = net(images)
            log_probs2 = pre_net(images)
            loss = self.loss_func(log_probs, labels).requires_grad_(True)    #
            loss.retain_grad()
            loss.backward()
            if batch_idx == 0:
                loss2 = self.loss_func(log_probs2, labels).requires_grad_(True)    #
                loss2.retain_grad()
                loss2.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        for k, v in pre_net.named_parameters():
            if k[-1] == 't':
                glob_w.append(v.data)
                glob_grad.append(v.grad)
        for k, v in net.named_parameters():
            if k[-1] == 't':
                tmp_w.append(v.data)
                tmp_grad.append(v.grad)
        norm1, norm2 = 0, 0
        # print('pre_w:', pre_w)
        # print('pre_grad:', pre_grad)
        # print('tmp_w:', tmp_w)
        # print('tmp_grad:', tmp_grad)
        if glob_w and glob_grad:
            for i in range(len(tmp_w)):
                norm1 += np.linalg.norm(tmp_grad[i] - glob_grad[i])
                norm2 += np.linalg.norm(tmp_w[i] - glob_w[i])
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        print('norm1: ', norm1, end='\t')
        print('norm2: ', norm2)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), tmp_w, tmp_grad, norm1/norm2 if norm2 != 0 else 0

def estimate():
    """Algorithm 2: Parameters estimation"""
    # L_max, L_avg = 33.94554781046159, 11.77769055780446
    # L = L_avg
    # mu = 47
    # L = (1539.6969388165412 ** 0.5)
    # mu = 24.274257548093622
    bigf = 0.05
    ci, cn, dif = 3.62, 4.56, 0.5
    for L in range(1, 60):    # [ci+dif, cn+dif, ci, cn]:
        for mu in range(1, 60):     # [3.45, 4.59, 3.75, 4.89]:
            deprecated = False
            tau, c, lr = {}, {}, {}
            lamb = 0.01  # MNIST
            tauMaxList, tauAvgList, yitaList, KList, tAvgList, XList = [], [], [], [], [], []    # for diff M
            for m in range(1, 11):
                t, tauList = [], []  # for diff round
                p = getPrepareTime(10)
                r = copy.deepcopy(p)
                for i in range(10):
                    tau[i], c[i], lr[i] = 0, 0, lamb
                tauMax, tau_th = 0, 999
                for k in range(100):
                    chosen, r, tmp_t = chooseClientsSemiAsync(m, p, r)
                    t.append(tmp_t)

                    for i in range(10):
                        if i in chosen:
                            tau[i] = 0
                        else:
                            tau[i] += 1
                    if max(tau.values()) > tauMax:
                        tauMax = max(tau.values())
                    tauList.append(sum(tau.values()) / len(tau.values()))

                    for i in chosen:
                        c[i] += 1
                    count_total = sum(c.values())
                    f = {}
                    for idx in chosen:
                        f[idx] = c[idx] / count_total
                    # if m == 10:
                    #     print("C", c)
                    #     print("F", f)
                    for idx in chosen:
                        frequency = f[idx]
                        true_learning_rate = lamb / (10 * frequency)
                        true_learning_rate = max(0.001, true_learning_rate)
                        lr[idx] = true_learning_rate
                # yita = max(lr.values())
                yita = sum(lr.values()) / len(lr.values())
                yitaList.append(yita)
                alpha = m / 10
                beta = alpha
                tAvg = sum(t) / len(t)
                tauAvg = sum(tauList) / len(tauList)
                # tauAvg = sum(tau.values()) / len(tau.values())
                # print(yita)
                # smallf = 1 - 2 * alpha * lamb * beta * (mu - yita * (L ** 2))
                smallf = 1 - 2 * lamb * (mu - yita * (L ** 2))
                try:
                    # X = math.log(bigf, smallf)
                    X = math.log(2 * bigf, smallf)
                    # X = math.log(4 * bigf, smallf)
                    XList.append(X)
                    K = (1 + tauAvg) * X
                    # K = (1 + tauMax) * X
                except:
                    deprecated = True
                    continue
                else:
                    tauMaxList.append(tauMax)
                    tauAvgList.append(tauAvg)
                    KList.append(round(K, 2))
                    tAvgList.append(tAvg)
            # if (not deprecated) and (KList[0] < 0 or KList[0] > 2000 or KList[9] < 30):
            #     deprecated = True
            if not deprecated: # and totalt[-1] > min(totalt):
                totalt = [round(KList[i] * tAvgList[i], 2) for i in range(10)]
                if not(totalt[-1] < 3e3 or totalt[0] > 1e5 or totalt[0] < 3e3):
                    print("L: ", L, "mu: ", mu)
                    # print("tauMax:", tauMaxList)
                    print("tauAvg:", tauAvgList)
                    print("yita:", yitaList)
                    print("X:", XList)
                    print("K:", KList)
                    print("t:", tAvgList)
                    print("Total t:", totalt, end='\n\n')
                    # file = open('./log/mu_L_trial2.txt', "a")
                    # file.write("L: {}, mu: {}".format(L, mu) + "\n")
                    # # file.write("tauMax:" + str(tauMaxList) + "\n")
                    # file.write("tauAvg:" + str(tauAvgList) + "\n")
                    # file.write("yita:" + str(yitaList) + "\n")
                    # file.write("X:" + str(XList) + "\n")
                    # file.write("K:" + str(KList) + "\n")
                    # file.write("t:" + str(tAvgList) + "\n")
                    # file.write("Total t:" + str(totalt) + "\n")
                    # file.write("\n\n\n")
                    # return yitaList[4], yitaList[9]

def get_L():
    """
    计算文章中L的值
    -> 计算梯度 -> 计算梯度的范式 -> 计算权重的范式 -> 相除
    """
    N = 10
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    # dict_users = mnist_iid(dataset_train, N)
    # dict_users = mnist_noniid(dataset_train, N)
    # dict_users = cifar_iid(dataset_train, N)
    dict_users = cifar_noniid(dataset_train, N)
    net_glob = MNIST_CNN_Net().to('cpu')
    net_glob.train()
    prepareTime = getPrepareTime(N)
    currentLostTime = copy.deepcopy(prepareTime)
    learning_rate = [0.01 for i in range(N)]
    runtime, rnd = 0, 1
    betaList = []
    while rnd < 10:
        w_locals, loss_locals = {}, []
        chosenClients, currentLostTime, iterationTime = \
            chooseClientsSemiAsync(N, prepareTime, currentLostTime)
        runtime += iterationTime
        beta = 0
        for k, v in net_glob.named_parameters():
            v.retain_grad()
        for idx in chosenClients:   # 本地更新
            local = LocalUpdate(dataset=dataset_train, idxs=dict_users[idx]
                                , learning_rate=learning_rate[idx])
            w, loss, tmp_w, tmp_grad, tmp_beta = local.train(copy.deepcopy(net_glob).to('cpu'))
            beta += tmp_beta
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        tmp_net_glob = copy.deepcopy(net_glob.state_dict())
        # if rnd > 1:
        betaList.append(beta/N)
        w_glob = FedSA(False, tmp_net_glob, w_locals, 1.0, [0 for i in range(N)], 999, copy.deepcopy(chosenClients))
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        print(rnd, betaList, end='\n\n')
        rnd += 1
    print(betaList)
    print('beta_avg', sum(betaList) / len(betaList))
    print('beta_min', min(betaList))
    print('beta_max', max(betaList))

def backforth_get_L():
    lamb = 0.01

    yita1 = 0.02265151515151515
    phi1 = 0.05 ** (1 / 55)
    alpha1 = 0.5
    beta1 = 0.1

    yita2 = 0.016566265060240965
    phi2 = 0.05 ** (1 / 120)
    alpha2 = 1.0
    beta2 = 0.1

    b1 = (1-phi1) / (2 * alpha1 * lamb * beta1)
    b2 = (1-phi2) / (2 * alpha2 * lamb * beta2)
    L2 = (b1 - b2) / (yita2 - yita1)
    print(L2)
    mu = (b1 + yita1 * L2 + b2 + yita2 * L2) / 2
    print(mu)

    L2 = -1539.6969388165412
    mu = -24.274257548093622


K = []

if __name__ == '__main__':
    estimate()
    # backforth_get_L()

    # mu, L = 4, 4
    # a = 0.05
    # # phi=1 - 2 * alpha * lambda * beta * (mu - eta * L^2)
    # b = 1 - 2 * 0.1 * 0.01 * 0.1 * (mu - 0.0141 * L * L)
    # x = math.log(a, b)
    # k = x * (5.90 + 1)
    # print("mu: ", mu, "L: ", L)
    # print("X: ", x)
    # print("K: ", k)

    # get_L()
    # np.array()

