#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:Hongxu_ICDM
@author:xiangguosun
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: user_item_representation.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1
@time: 2020/05/20
"""
import torch
import torch.nn as nn
import os
import pandas as pd
from collections import defaultdict
import pickle
import joblib

#dataset_name = 'Amazon_Musical_Instruments'
#dataset_name = 'Amazon_Automotive'
dataset_name = 'Amazon_Toys_Games'
#dataset_name = 'Amazon_CellPhones_Accessories'
#dataset_name = 'Amazon_Grocery_Gourmet_Food'
#dataset_name = 'Amazon_Books'
#dataset_name = 'Amazon_CDs_Vinyl'


embedding_size = 100

if __name__ == '__main__':
#def user_item_representation(dataset_name):
    
    print('-'*100)
    print(f'{dataset_name}......')
    print('-'*100)

    folder = f'../data/{dataset_name}/'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print(f'user_item_representation.py device: {device}')

    # Writing our model
    class Autoencoder(nn.Module):
        def __init__(self, d_in=2000, d_hid=800, d_out=embedding_size):
            super(Autoencoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(d_in, d_hid),
                nn.ReLU(True),
                nn.Linear(d_hid, d_out),
                nn.ReLU(True))

            self.decoder = nn.Sequential(
                nn.Linear(d_out, d_hid),
                nn.ReLU(True),
                nn.Linear(d_hid, d_in),
                nn.ReLU(True))

        def forward(self, x):
            self.embeddings = self.encoder(x)
            print(self.embeddings.shape)
            xx = self.decoder(self.embeddings)
            print(xx.shape)
            return xx

        def save_embeddings(self):
            return self.embeddings

    user_item_relation = pd.read_csv(folder + 'user_item.relation', header=None, sep=',')
    """
    construct adj：user_item
    """
    user_id2local_id = defaultdict(int)
    item_id2local_id = defaultdict(int)
    users = set(user_item_relation[0])
    items = set(user_item_relation[1])
    user_global_id_sequence = []
    for i, uid in enumerate(users):
        user_id2local_id[uid] = i
        user_global_id_sequence.append(uid)
    for i, iid in enumerate(items):
        item_id2local_id[iid] = i

    print('users:', len(users), 'items:', len(items))

    adj = torch.zeros((len(users), len(items)))
    for _, row in user_item_relation.iterrows():
        adj[user_id2local_id[row[0]], item_id2local_id[row[1]]] = 1.0
    print(f'adj.shape={adj.shape}')
    model = Autoencoder(d_in=adj.shape[1], d_hid=800, d_out=embedding_size).to(device)
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00005)
    num_epochs = 10

    for epoch in range(num_epochs):
        output = model(adj.to(device))
        loss = distance(output.to(device), adj.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {}/{}, loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    embeddings = model.save_embeddings()
    print(f'embeddings:{embeddings.shape}')

    user_item_representation = defaultdict(torch.Tensor) #user的embedding
    for i, user in enumerate(user_global_id_sequence):
        user_item_representation[user] = embeddings[i, :]

    #pickle.dump(user_item_representation, open(folder + 'user_item_dic.wv', 'wb'))
    with open(folder + 'user_item_dic.wv', 'wb') as fo :
        joblib.dump(user_item_representation,fo)
    print('save done!')
