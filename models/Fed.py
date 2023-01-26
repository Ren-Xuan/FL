#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from collections import OrderedDict
import copy
import math
from typing import Dict, List
import numpy as np
import torch
from torch import nn


def FedAvg(w:Dict):
    userIdx = list(w.keys())
    w_avg = copy.deepcopy(w[userIdx[0]])
    for k in w_avg.keys():
        for i in userIdx[1:]:
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def FedSlice(userWeightDict:Dict,indexWeight:Dict = None,generateGraph = False,randFlag = False):
    if not randFlag:
        userIdx = list(userWeightDict.keys())
        num_users = len(userIdx)
        w_avg = OrderedDict()
        for k in userWeightDict[userIdx[0]].keys():
            w_avg[k] = torch.zeros_like(userWeightDict[userIdx[0]][k])
        for k in w_avg.keys():
            if len(w_avg[k].size()) == 1:
                idx_shard = [i for i in range(w_avg[k].size()[0])]
                choiceOnce = w_avg[k].size()[0]//num_users + 1 if w_avg[k].size()[0]%num_users != 0 else w_avg[k].size()[0]//num_users
                choiceOnce+=1
                for i in userIdx:
                    if len(idx_shard)<=choiceOnce:#when available choice less than choiceOnce 
                        rand_set = idx_shard
                    else:
                        if randFlag:
                            rand_set = set(np.random.choice(idx_shard,choiceOnce, replace=False))
                        else:
                            rand_set = set(idx_shard[:choiceOnce])
                        idx_shard = list(set(idx_shard) - rand_set)
                    for idx in rand_set:
                        w_avg[k][idx] = userWeightDict[i][k][idx]
            elif len(w_avg[k].size()) == 2:
                idx_dic = []
                num_shards = 0
                for i in range(w_avg[k].size()[0]):
                    for j in range(w_avg[k].size()[1]):
                        idx_dic.append((i,j))
                        num_shards+=1
                idx_shard = [i for i in range(len(idx_dic))]
                choiceOnce = num_shards//num_users +1 if num_shards%num_users!=0 else num_shards//num_users
                for i in userIdx:
                    if len(idx_shard)<=choiceOnce:
                        rand_set = idx_shard
                    else:
                        if randFlag:
                            rand_set = set(np.random.choice(idx_shard,choiceOnce, replace=False))
                        else:
                            rand_set = set(idx_shard[:choiceOnce])
                        idx_shard = list(set(idx_shard) - rand_set)
                    for idx in rand_set:
                        idx0 = idx_dic[idx][0]
                        idx1 = idx_dic[idx][1]
                        w_avg[k][idx0][idx1] = userWeightDict[i][k][idx0][idx1]
            elif len(w_avg[k].size()) == 4:
                idx_dic = []
                num_shards = 0
                for firstDegree in range(w_avg[k].size()[0]):
                    for secondDegree in range(w_avg[k].size()[1]):
                        for i in range(w_avg[k].size()[2]):
                            for j in range(w_avg[k].size()[3]):
                                idx_dic.append((firstDegree,secondDegree,i,j))#two dimention to one dimention 
                                num_shards+=1
                idx_shard = [i for i in range(len(idx_dic))]
                choiceOnce = num_shards//num_users +1 if num_shards%num_users!=0 else num_shards//num_users
                for i in userIdx:
                    if len(idx_shard)<=choiceOnce:
                        rand_set = idx_shard
                    else:
                        if randFlag:
                            rand_set = set(np.random.choice(idx_shard,choiceOnce, replace=False))
                        else:
                            rand_set = set(idx_shard[:choiceOnce])
                        idx_shard = list(set(idx_shard) - rand_set)
                    for idx in rand_set:
                        idx0 = idx_dic[idx][0]
                        idx1 = idx_dic[idx][1]
                        idx2 = idx_dic[idx][2]
                        idx3 = idx_dic[idx][3]
                        w_avg[k][idx0][idx1][idx2][idx3] = userWeightDict[i][k][idx0][idx1][idx2][idx3]
        return w_avg
def get_corr(X:torch.Tensor, Y:torch.tensor)->torch.tensor:
        X, Y = X.reshape(-1), Y.reshape(-1)
        X_mean, Y_mean = torch.mean(X), torch.mean(Y)
        corr = (torch.sum((X - X_mean) * (Y - Y_mean))) / (
                    torch.sqrt(torch.sum((X - X_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)).add(0.0000000001))
        return corr
