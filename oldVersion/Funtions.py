import copy
import math
import random
import time

import torch
import torch.nn as nn
import syft
import torch.optim as optim
from torch.autograd import Variable

import utils


def distributeData(hook, args, clientNum, trainLoader):
    client = {}
    clientList = []
    for i in range(1, clientNum + 1):
        client[str(i)] = syft.VirtualWorker(hook, id='client' + str(i))
        clientList.append(str(i))
    secureClient = syft.VirtualWorker(hook, id="secureClient")
    server = syft.VirtualWorker(hook, id="server")

    trainData, trainTarget = {}, {}
    for i in range(1, clientNum + 1):
        trainData[str(i)], trainTarget[str(i)] = [], []

    labelClusters = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))    # 将数据分为iid的群集
    # labelClusters = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))  # 将数据分为N-iid的群集
    clustersNum = len(labelClusters)  # 上述群集的数量
    clustersDataNum, clustersClientNum = {}, {}
    for i in range(1, clientNum + 1):
        cluster = i % clustersNum
        clustersClientNum.setdefault(cluster, [])
        clustersClientNum[cluster].append(i)

    for i, (X, label) in enumerate(trainLoader):
        # N-iid
        # a_label = label.numpy().tolist()[0]
        # for j in range(clustersNum):
        #     if a_label in labelClusters[j]:
        #         cluster = j  # 指的是标签集的index
        #         clustersDataNum.setdefault(cluster, -1)
        #         clustersDataNum[cluster] += 1
        #         clientOrder = clustersDataNum[cluster] % len(clustersClientNum[cluster])
        #         chooseClient = clustersClientNum[cluster][clientOrder]
        #         # distribute
        #         X = Variable(X)
        #         label = Variable(label)
        #         if len(X) != args.batchSize:
        #             continue
        #         trainData[str(chooseClient)].append(X.numpy().tolist())
        #         trainTarget[str(chooseClient)].append(label.numpy().tolist())
        # iid
        X = Variable(X)
        label = Variable(label)
        if len(X) != args.batchSize:
            continue
        trainData[str(i % clientNum + 1)].append(X.numpy().tolist())
        trainTarget[str(i % clientNum + 1)].append(label.numpy().tolist())

    clientData = {}
    clientTarget = {}
    for i in range(1, clientNum + 1):
        trainData[str(i)] = torch.tensor(trainData[str(i)])
        trainTarget[str(i)] = torch.tensor(trainTarget[str(i)])
        clientData[str(i)] = trainData[str(i)].send(client[str(i)])
        clientTarget[str(i)] = trainTarget[str(i)].send(client[str(i)])
    return client, secureClient, server, clientData, clientTarget


def getPrepareTime(clientNum):
    lowerBound, upperBound = 10, 100
    prepareTime = {}
    nowValue = lowerBound
    differenceValue = (upperBound - lowerBound) / (clientNum - 1)
    for i in range(1, clientNum + 1):
        prepareTime[str(i)] = nowValue + random.uniform(-0.0001, 0.0001)
        nowValue += differenceValue
    return prepareTime


def chooseClients(M, prepareTime, currentLostTime):
    sortedCurrentLostTime = sorted(currentLostTime.items(), key=lambda x: x[1], reverse=False)
    chosenClients = []
    for i in range(0, M):
        chosenClients.append(sortedCurrentLostTime[i][0])  # 选中所需时间较小的前M个
    iterationTime = sortedCurrentLostTime[M - 1][1]  # 该轮完成所需的时间为M个客户里所需的最长时间

    for client in list(currentLostTime.keys()):  # 更新所有客户的仍需等待的时间
        if currentLostTime[client] <= iterationTime:
            currentLostTime[client] = prepareTime[client]
        else:
            currentLostTime[client] -= iterationTime

    return chosenClients, currentLostTime, iterationTime


def localUpdate(device, clients, clientData, clientOpt, clientModel, clientTarget, lossF):
    for client in clients:
        for wi in range(1):
            for j in range(0, len(clientData[str(client)])):
                clientOpt[str(client)].zero_grad()
                out = clientModel[str(client)](clientData[str(client)][j].to(device))
                label = clientTarget[str(client)][j].to(device)
                lossvalue = lossF(out, label)
                lossvalue.backward()  # 反向转播，刷新梯度值
                clientOpt[str(client)].step()
    return clientData, clientOpt, clientModel, clientTarget, lossF


def add_model(dst_model, src_model):
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] + params2[name1]
                # print(type(params1[name1]))
                # print(params1[name1])
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model


def scale_model(model, scale):
    params = model.state_dict().copy()
    # scale = torch.tensor(scale)
    with torch.no_grad():
        for name in params:
            # params[name] = params[name].type_as(scale) * scale
            params[name] = params[name] * scale
    scaled_model = copy.deepcopy(model)
    scaled_model.load_state_dict(params, strict=False)
    return scaled_model


def aggregate_model(server_model, update_model, yita):
    params1 = server_model.state_dict().copy()
    params2 = update_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = (1 - yita) * params1[name1] + yita * params2[name1]
    model = copy.deepcopy(server_model)
    model.load_state_dict(params1, strict=False)
    return model


def globalAggregate(serverModel, clients, clientNum, clientModel, tau, tau_th):
    updateWorkerNum = 0
    with torch.no_grad():
        for client in clients:
            if tau[str(client)] > tau_th:
                continue
            if updateWorkerNum == 0:
                updateModel = clientModel[str(client)].copy()
            else:
                updateModel = add_model(updateModel, clientModel[str(client)])
            updateWorkerNum += 1
        if updateWorkerNum != 0:
            updateModel = scale_model(updateModel, 1.0 / updateWorkerNum)
            model = aggregate_model(serverModel, updateModel, updateWorkerNum / clientNum)
            del updateModel
        else:
            model = scale_model(serverModel, 1)
    return model


# def distributeModel(client, clients, clientModel, clientOpt):
#     for i in clients:
#         clientModel[str(i)] = model.copy().send(client[str(i)])
#         clientOpt[str(i)] = optim.SGD(params=clientModel[str(i)].parameters(), lr=args.lr)
#     return client, clients, clientModel, clientOpt


def modelTest(ver, device, args, model, runtime, worker_num, test_loader):
    loss_f = nn.CrossEntropyLoss()
    model.eval()
    now_loss, correct = 0, 0
    model = model.get()
    for i, (X, label) in enumerate(test_loader):
        X = X.to(device)
        label = label.to(device)
        X = Variable(X)
        testout = model(X)
        testloss = loss_f(testout, label)
        pred = testout.argmax(1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        now_loss += float(testloss)
    now_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.
          format(now_loss, correct, len(test_loader) * args.batchSize,
                 100. * correct / (len(test_loader) * args.batchSize)))

    f = open(ver + args.modelType + "_" + str(
        args.batchSize) + "_" + str(args.lr) + "_" + str(worker_num) + ".txt", 'a')
    newline = str(now_loss) + "	" + str(correct / (len(test_loader) * args.batchSize)) + "	" + str(runtime)
    f.writelines(newline + "\n")
    f.close()
    return 100 * correct / (len(test_loader) * args.batchSize)


# Joint DP

def localDP(device, clientModel, choseClients):
    noisyModel = copy.deepcopy(clientModel)
    for client in choseClients:
        noisyLocal = noisyModel[str(client)].state_dict().copy()
        # print(noisyLocal)
        with torch.no_grad():
            for k in noisyLocal.keys():
                # noisyLocal[k] = gaussianMechanism(device, noisyLocal[k])
                gaussianMechanism(device, noisyLocal[k])
    # model = copy.deepcopy(dst_model)
    # model.load_state_dict(params1, strict=False)
    return noisyModel


def gaussianMechanism(device, x, df=.1, eps=1, delta=0.1):
    # clients specify sensitivity df
    std = (2 * math.log(1.25 / delta)) ** 0.5 * df / eps
    noise = torch.normal(mean=0.0, std=std, size=x.size()).to(device)
    # noise = torch.tensor(mean=0.0, std=std, size=x.size()).to(device)
    print(type(noise))
    print(noise)
    # return x+noise
