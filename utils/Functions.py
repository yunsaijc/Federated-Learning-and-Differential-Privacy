import random

import numpy as np


def getPrepareTime(clientNum):
    lowerBound, upperBound = 10, 100
    prepareTime = {}
    nowValue = lowerBound
    differenceValue = (upperBound - lowerBound) / (clientNum - 1)
    for i in range(clientNum):
        prepareTime[i] = nowValue + random.uniform(-0.0001, 0.0001)
        nowValue += differenceValue
    # for i in range(clientNum):
    #     prepareTime[i] = random.randint(10, 100)
    return prepareTime


def chooseClientsSync(num_users, m, prepareTime, currentLostTime):
    chosenClients = np.random.choice(range(num_users), m, replace=False)
    iterationTime = max([prepareTime[i] for i in chosenClients])
    for client in list(currentLostTime.keys()):  # 更新所有客户的仍需等待的时间
        if currentLostTime[client] <= iterationTime:
            currentLostTime[client] = prepareTime[client]
        else:
            currentLostTime[client] -= iterationTime
    return chosenClients, currentLostTime, iterationTime


def chooseClientsSemiAsync(M, prepareTime, currentLostTime):
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


def chooseClientsAsync(M, prepareTime, currentLostTime):
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
