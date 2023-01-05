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
    if len(chosen) != 0:
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
    else:
        w_result = copy.deepcopy(former)
        w_avg = copy.deepcopy(former)
    # return w_avg
    return w_result
