#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM, MNIST_CNN_Net, CIFAR_CNN_Net
from models.Fed import FedAvg, FedSA
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.Functions import *

matplotlib.use('Agg')

if __name__ == '__main__':
    # parse args
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 判断是否需要自动遍历，从而设置M和N的上下界
    if args.allParams:
        if args.part % 10 == 0:
            frac_min = 1.0
        else:
            frac_min = round(args.part % 10 * 0.1, 1)
        frac_max = 1.01
        N_min = args.sec * 100 + (args.part // 10 + 1) * 10
        N_max = (args.sec + 1) * 100 + 1
    else:
        frac_min, frac_max = args.frac, args.frac + 0.01
        N_min, N_max = args.num_users, args.num_users + 1

    if args.sync:
        args.selfLR = False

    for num_users in range(N_min, N_max, 10):
        if num_users == N_min:
            frac = frac_min
        else:
            frac = 0.1
        while frac <= frac_max:
        # for frac in np.arange(frac_min, frac_max, step=0.1).round(1):
            print("N = {}, M = {} * N-----------------------------------------------".format(num_users, frac))

            # load dataset and split users
            if args.dataset == 'mnist':
                trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
                dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
                args.num_channels = 1
                # sample users
                if args.iid:
                    dict_users = mnist_iid(dataset_train, num_users)
                else:
                    dict_users = mnist_noniid(dataset_train, num_users)
            elif args.dataset == 'cifar':
                # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                # 0.5))])
                args.num_channels = 3
                trans_cifar_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                trans_cifar_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
                dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
                if args.iid:
                    dict_users = cifar_iid(dataset_train, num_users)
                else:
                    dict_users = cifar_noniid(dataset_train, num_users)
            elif args.dataset == 'fashion-mnist':
                args.num_channels = 1
                trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                                      transform=trans_fashion_mnist)
                dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                                     transform=trans_fashion_mnist)
                if args.iid:
                    dict_users = mnist_iid(dataset_train, num_users)
                else:
                    dict_users = mnist_noniid(dataset_train, num_users)
            elif args.dataset == 'femnist':
                args.num_channels = 1
                dataset_train = FEMNIST(train=True)
                dataset_test = FEMNIST(train=False)
                dict_users = dataset_train.get_client_dic()
                num_users = len(dict_users)
                if args.iid:
                    exit('Error: femnist dataset is naturally non-iid')
                else:
                    print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
            elif args.dataset == 'shakespeare':
                dataset_train = ShakeSpeare(train=True)
                dataset_test = ShakeSpeare(train=False)
                dict_users = dataset_train.get_client_dic()
                num_users = len(dict_users)
                if args.iid:
                    exit('Error: ShakeSpeare dataset is naturally non-iid')
                else:
                    print("Warning: The ShakeSpeare dataset is naturally non-iid, "
                          "you do not need to specify iid or non-iid")
            else:
                sys.exit('Error: unrecognized dataset')
            img_size = dataset_train[0][0].shape

            # build model
            if args.model == 'cnn' and args.dataset == 'cifar':
                # net_glob = CNNCifar(args=args).to(args.device)
                net_glob = CIFAR_CNN_Net().to(args.device)
            elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
                # net_glob = CNNMnist(args=args).to(args.device)
                net_glob = MNIST_CNN_Net().to(args.device)
            elif args.dataset == 'femnist' and args.model == 'cnn':
                net_glob = CNNFemnist(args=args).to(args.device)
            elif args.dataset == 'shakespeare' and args.model == 'lstm':
                net_glob = CharLSTM().to(args.device)
            elif args.model == 'mlp':
                len_in = 1
                for x in img_size:
                    len_in *= x
                net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
            else:
                sys.exit('Error: unrecognized model')

            # dp settings
            dp_epsilon = args.dp_epsilon / (frac * args.epochs)     # 将全局隐私预算转换为本地隐私预算
            # dp_epsilon = 0.05
            dp_delta = args.dp_delta
            dp_mechanism = args.dp_mechanism
            dp_clip = args.dp_clip      # 梯度裁剪的参数C

            print(net_glob)
            net_glob.train()

            # copy weights
            w_glob = net_glob.state_dict()  # 获取当前网络的权重矩阵
            all_clients = list(range(num_users))

            # initialize
            prepareTime = getPrepareTime(num_users)     # 模型准备时间p
            # print(prepareTime)
            currentLostTime = copy.deepcopy(prepareTime)    # 完成下一轮本地训练仍需要的时间r
            acc_test, loss_test, realtimeList = {}, {}, {}  # 准确率，loss值，代码实际运行时间的字典
            global_learning_rate = args.lr                  # 全局学习率
            learning_rate = [args.lr for i in range(num_users)]
            runtime, realtime = 0, 0    # 运行时间(根据模型准备时间计算而来)，实际运行时间(实际计时的结果)
            tauList = []        # 存放每一轮全局训练中tau的最大值
            tauMax = 0          # 初始化tau的最大值
            T_avg_list = []     # 存放每一轮全局训练中所需的平均运行时间
            rnd = 1             # 初始化全局轮数
            count = {}          # 记录每个客户参与全局训练的次数
            tau = {}            # 模型陈旧度
            tau_th = 999        # 模型陈旧度的阈值
            for i in range(num_users):
                tau[i] = 0
                count[i] = 0

            # while rnd < args.epochs:
            # while realtime < 5000:
            while runtime < 5000:
                time_start = time.time()

                if args.asy:
                    m = 1
                    # frac = 0.1
                    frac = m / args.num_users
                else:
                    m = max(int(frac * num_users), 1)

                # 选择这一轮参与全局训练的客户端
                if args.sync:  # 同步, FedAvg
                    w_locals, loss_locals = [], []
                    chosenClients, currentLostTime, iterationTime = chooseClientsSync(num_users,
                                                                                      m, prepareTime, currentLostTime)
                elif (not args.sync) and args.avg:  # 半异步，但以FedAvg方式聚合
                    w_locals, loss_locals = [], []
                    chosenClients, currentLostTime, iterationTime = \
                        chooseClientsSemiAsync(m, prepareTime, currentLostTime)
                elif args.asy:                      # 异步，以FedSA方式聚合
                    w_locals, loss_locals = {}, []
                    chosenClients, currentLostTime, iterationTime = \
                        chooseClientsAsync(m, prepareTime, currentLostTime)
                else:     # 半异步
                    w_locals, loss_locals = {}, []
                    chosenClients, currentLostTime, iterationTime = \
                        chooseClientsSemiAsync(m, prepareTime, currentLostTime)

                runtime += iterationTime
                T_avg_list.append(round(runtime / rnd, 3))

                # local update
                for idx in chosenClients:
                    count[idx] += 1
                    # args.lr = learning_rate[idx]
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                        dp_epsilon=dp_epsilon, dp_delta=dp_delta, dp_mechanism=dp_mechanism,
                                        dp_clip=dp_clip, learning_rate=learning_rate[idx])
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    # learning_rate[idx] = curLR
                    if args.sync or args.avg:
                        w_locals.append(copy.deepcopy(w))
                    else:
                        w_locals[idx] = copy.deepcopy(w)
                    loss_locals.append(copy.deepcopy(loss))

                # update adaptive learning rate
                if args.selfLR:
                    count_total = sum(count.values())
                    f = {}
                    for idx in chosenClients:
                        f[idx] = count[idx] / count_total
                    for idx in chosenClients:
                        frequency = f[idx]
                        true_learning_rate = global_learning_rate / (num_users * frequency)
                        true_learning_rate = max(0.001, true_learning_rate)
                        learning_rate[idx] = true_learning_rate

                # global update
                if args.sync or args.avg:
                    w_glob = FedAvg(w_locals)
                else:
                    tmp_net_glob = copy.deepcopy(net_glob.state_dict())
                    w_glob = FedSA(args.sync, tmp_net_glob, w_locals, frac, tau, tau_th, copy.deepcopy(chosenClients))
                net_glob.load_state_dict(w_glob)    # copy weight to net_glob

                # update tau
                for i in range(num_users):
                    if i in chosenClients:
                        tau[i] = 0
                    else:
                        tau[i] += 1
                if max(tau.values()) > tauMax:
                    tauMax = max(tau.values())
                tauList.append(tauMax)
                # print()
                # print(chosenClients)
                # print("tau:"+ str(tau))
                # print(tauMax)

                # print accuracy
                net_glob.eval()
                acc_t, loss_t = test_img(net_glob, dataset_test, args)

                time_end = time.time()
                realtime += time_end - time_start

                print("Round {:3d},Testing accuracy: {:.4f}, Loss: {:.4f}, runtime: {:.4f}, realtime: {:.4f}".
                      format(rnd, acc_t, loss_t, runtime, realtime))
                rnd += 1
                acc_test[runtime] = acc_t       # .item()
                loss_test[runtime] = loss_t     # .item()
                realtimeList[runtime] = realtime     # .item()

            # 将结果写入文件
            rootpath = './log'
            if not os.path.exists(rootpath):
                os.makedirs(rootpath)
            if args.sync:
                mechanism = "Sync"
            else:
                mechanism = "Semi"
            accfile = open(rootpath + '/' + mechanism + '_{}_{}_{}_{}_{}_dp_{}_epsilon_{}_{}_{}.txt'.
                           format(args.dataset, num_users, args.model, args.epochs,
                                  args.dp_mechanism, args.dp_epsilon, args.iid, frac, tau_th), "w")
            for key in acc_test.keys():
                sac = "{:4f}\t\t{:4f}\t\t{:4f}\t\t{:4f}".\
                    format(key, float(acc_test[key]), loss_test[key], realtimeList[key])
                accfile.write(sac)
                accfile.write('\n')
            accfile.close()

            # 将陈旧度写入文件
            # tau_file = open(rootpath + '/tau_{}_{}_{}_{}_{}_dp_{}_epsilon_{}_{}.txt'.
            #                format(args.dataset, num_users, args.model, args.epochs,
            #                       args.dp_mechanism, args.dp_epsilon, args.iid, frac), "w")
            # for ind in range(rnd-1):
            #     sac = "{:d}\t\t{:d}\t\t{:3f}".format(ind+1, tauList[ind], T_avg_list[ind])
            #     tau_file.write(sac)
            #     tau_file.write('\n')
            # tau_file.close()

            frac += 0.1
            frac = round(frac, 1)

            # plot loss curve
            # plt.figure()
            # plt.plot(acc_test.keys(), acc_test.values())
            # plt.ylabel('test accuracy')
            # plt.savefig(rootpath + '/' + str(args.sync) + '_fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_{}_acc.png'.format(
            #     args.dataset, args.model, args.epochs, frac, args.iid, args.dp_mechanism, args.dp_epsilon, frac))
