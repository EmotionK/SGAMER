#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:deepwalk-master
@author:xiangguosun
@contact:sunxiangguodut@qq.com
@website:http://blog.csdn.net/github_36326955
@file: embed_nodes.py
@platform: macOS High Sierra 10.13.1 Pycharm pro 2017.1
@time: 2018/09/13
"""



import random
from gensim.models import Word2Vec
import pickle
import torch
from collections import defaultdict
from model.util import simple_walks as serialized_walks

def embedding_category_brand(dataset_name,user_number,item_number):
    number_walks = 10
    walk_length = 6  # length of path
    workers = 2
    representation_size = 192
    window_size = 3
    output = f'../data/{dataset_name}/node.wv'
    G = pickle.load(open(f'../data/{dataset_name}/graph.nx', 'rb'))  # node 包括 user/item/brand/category/also_bought
    walks_filebase = f'../data/{dataset_name}/walks.txt'
    nodewv = f'../data/{dataset_name}/nodewv.dic'
    print("Number of nodes: {}".format(G.number_of_nodes()))
    print("Number of edges: {}".format(G.number_of_edges()))
    print("number_walks: {}".format(number_walks))
    num_walks = G.number_of_nodes() * number_walks
    print("Number of walks: {}".format(num_walks))
    data_size = num_walks * walk_length
    print("Data size (walks*length): {}".format(data_size))

    print(type(G))

    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                      path_length=walk_length, num_workers=workers, alpha=0.1,
                                                      rand=random.Random(100), always_rebuild=True)
    walks = serialized_walks.WalksCorpus(walk_files)

    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
                     workers=workers)

    model.wv.save_word2vec_format(output)

    nodewv_dic = defaultdict(torch.Tensor)
    with open(output, 'r') as f: #只要类别和商标的embedding
        f.readline()
        for line in f:
            s = line.split()
            nodeid = int(s[0])
            if nodeid < 10479:
                continue
            fea = [float(x) for x in s[1:]]
            nodewv_dic[nodeid] = torch.Tensor(fea)

    i = 0
    user_embedding = pickle.load(open(f'../data/{dataset_name}/torch.user_embedding','rb'))
    user_embedding = torch.reshape(user_embedding,(-1,192))
    print(f'user_embedding_shape:{user_embedding.shape}')
    for j in range(user_number):
        nodewv_dic[i] = torch.Tensor(user_embedding[j])
        i += 1

    item_embedding = pickle.load(open(f'../data/{dataset_name}/torch.item_embedding', 'rb'))
    print(f'item_embedding_shape:{item_embedding.shape}')
    for k in range(item_number):
        item_embedding_list = item_embedding[k].tolist()
        item_embedding_fill = item_embedding_list + item_embedding_list + item_embedding_list
        nodewv_dic[i] = torch.Tensor(item_embedding_fill)
        i += 1

    pickle.dump(nodewv_dic, open(nodewv, 'wb'))
    print(len(nodewv_dic))
