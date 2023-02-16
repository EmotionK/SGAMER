#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:Hongxu_ICDM
@file: meta_path_instances_representation.py
@time: 2020/06/08
"""

import sys
import os
sys.path.append('..')

from gensim.models import Word2Vec
from model.util.data_utils import *
import torch
from torch import nn
import numpy as np
import pickle

"""
ui_path_vectors:{

}
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embedding_size = 100

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
        xx = self.decoder(self.embeddings)
        return xx

    def save_embeddings(self):
        return self.embeddings

def instance_emb(metapath_file, output_file):
    walks = get_instance_paths(metapath_file) #路径列表[['95', '10553', '11619', '6611'], ['95', '10553', '11619', '6611'], ....]
    path_dict = instance_paths_to_dict(metapath_file) #{(95, 6611): [['95', '10553', '11619', '6611'], ['95', '10553', '11619', '6611']], (95, 3241): [['95', '10553', '11619', '3241']],...}

    print("Training...")
    model = Word2Vec(walks, size=embedding_size, window=3, min_count=0, sg=1, hs=1,
                     workers=1)

    # mean pooling
    ui_path_vectors = {}
    for ui, ui_paths in path_dict.items():#ui:(95, 6611)---ui_paths:[['95', '10553', '11619', '6611'], ['95', '10553', '11619', '6611']]
        for path in ui_paths:#['95', '10553', '11619', '6611']
            nodes_vectors = []
            for nodeid in path:
                nodes_vectors.append(model.wv[nodeid])#model.wv[nodeid]输出nodeid的特征映射结果
            nodes_np = np.array(nodes_vectors)#nodes_np二维向量数组
            path_vector = np.mean(nodes_np, axis=0) #按列求均值,100size的一维数组
            if ui not in ui_path_vectors.keys():
                ui_path_vectors[ui] = [path_vector]
                #print(f'path_vector:{type(path_vector)}') numpy.ndarray
            else:
                ui_path_vectors[ui].append(path_vector)
    pickle.dump(ui_path_vectors, open(output_file, 'wb'))


#dataset_name = 'Amazon_Musical_Instruments'
#dataset_name = 'Amazon_Automotive'
dataset_name = 'Amazon_Toys_Games'
#dataset_name = 'Amazon_CellPhones_Accessories'
#dataset_name = 'Amazon_Grocery_Gourmet_Food'
#dataset_name = 'Amazon_Books'
#dataset_name = 'Amazon_CDs_Vinyl'


if __name__ == '__main__':
#def meta_path_instances_representation(dataset_name):
    
    print('-'*100)
    print(f'{dataset_name}......')
    print('-'*100)
    
    folder = f'../data/{dataset_name}/'

    ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    ii_metapaths_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
    # ii form
    # embed ui paths
    for metapath in ui_metapaths_list:
        metapath_file = folder + metapath + '.paths'
        output_file = folder + metapath + '.wv'
        instance_emb(metapath_file, output_file)
    # embed ii paths
    ii_instance_file = folder + 'ii_random_form.paths'
    output_ii_emb_file = folder + 'ii_random_form.wv'
    # we randomly select 1 path from 7 item-item instances to generate 'this ii_random_form.wv', and then the following attention
    instance_emb(ii_instance_file, output_ii_emb_file)


