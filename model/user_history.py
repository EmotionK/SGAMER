#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:Hongxu_ICDM
@author:xiangguosun
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: user_history.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1
@time: 2020/05/20

input:
user_history.txt，超过N个以上的，选取前N个，剩下的留作测试集，小于N个的，直接pass掉，
item_item.wv
node.wv (user.feature,item.feature)

output:
user_id: features(2N-1,100)
training_links,
testing_links
训练集合，测试集合
"""
import torch
import pickle
from collections import defaultdict


dataset_name='Amazon_Musical_Instruments'

if __name__ == '__main__':
#def user_history(dataset_name):

    folder = f'../data/{dataset_name}/'
    bridge = 2
    train = 4
    test = 6

    history = []
    testing = []
    training = []

    '''
    """
    把nodewv转化为dic,或者是tensor
    """
    nodewv = folder + 'node.wv'

    nodewv_dic = defaultdict(torch.Tensor)
    with open(nodewv, 'r') as f:
        f.readline()
        for line in f:
            s = line.split()
            nodeid = int(s[0])
            fea = [float(x) for x in s[1:]]
            nodewv_dic[nodeid] = torch.Tensor(fea)

    print("node.feature done")
    print(len(nodewv_dic))  # 26333
    '''


    user_history_edges2id = pickle.load(open(folder + 'user_history.edges2id', 'rb'))

    """
    转化为 dic,tensor
    """

    ##################################
    item_item_wv_dic = defaultdict(torch.Tensor)
    with open(folder + 'item_item.wv', 'r') as f:
        f.readline()
        for line in f:
            s = line.split()
            item_item_id = int(s[0])
            fea = [float(x) for x in s[1:]]
            item_item_wv_dic[item_item_id] = torch.Tensor(fea)
    print("item_item.feature done")
    print(len(item_item_wv_dic))

    user_history_wv = defaultdict(torch.Tensor)
    with open(folder + 'user_history.txt', 'r') as f:
        for line in f:
            s = line.split()
            uid = int(s[0])
            item_history = [int(x) for x in s[1:]]
            if len(item_history) < (bridge + train):
                continue
            else:
                # 划分 train 和 test
                training.append([uid] + item_history[bridge:(bridge + train)])
                testing.append([uid] + item_history[bridge + train:])

    pickle.dump(training, open(folder + 'training', 'wb'))
    pickle.dump(testing, open(folder + 'testing', 'wb'))
    pickle.dump(user_history_wv, open(folder + 'user_history.wv', 'wb'))
