#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedSA(sync, former, w, frac, tau, tau_th, chosen: list):
    if not sync:
        for i in chosen:
            if tau[i] > tau_th:
                chosen.remove(i)
    w_avg = copy.deepcopy(w[chosen[0]])
    w_result = copy.deepcopy(former)
    valid = len(chosen)
    for k in w_avg.keys():
        for i in range(1, len(chosen)):
            w_avg[k] += w[chosen[i]][k]
        w_avg[k] = torch.div(w_avg[k], valid)
    for k in w_result.keys():
        for i in range(len(w)):
            if k in w_avg.keys():
                w_result[k] = (1 - frac) * w_result[k] + frac * w_avg[k]
    return w_avg


def FedSA_Agg(serverModel, clientNum, clients, clientModel, tau, tau_th):
    updateWorkerNum = 0
    for client in clients:
        if tau[client] > tau_th:
            continue
        if updateWorkerNum == 0:
            updateModel = clientModel[client].copy()
        else:
            updateModel = add_model(updateModel, clientModel[client])
        updateWorkerNum += 1
    if updateWorkerNum != 0:
        updateModel = scale_model(updateModel, 1.0 / updateWorkerNum)
        model = aggregate_model(serverModel, updateModel, updateWorkerNum / clientNum)
        del updateModel
    else:
        model = serverModel
    return model


def add_model(dst_model, src_model):
    params1 = dst_model.copy()
    params2 = src_model.copy()
    for name1 in params1:
        if name1 in params2:
            params1[name1] = params1[name1] + params2[name1]
    return copy.deepcopy(params1)


def scale_model(model, scale):
    params = model.copy()
    for name in params:
        params[name] = params[name] * scale
    return copy.deepcopy(params)


def aggregate_model(server_model, update_model, yita):
    params1 = server_model.copy()
    params2 = update_model.copy()
    for name1 in params1:
        if name1 in params2:
            params1[name1] = (1 - yita) * params1[name1] + yita * params2[name1]
    return copy.deepcopy(params1)
