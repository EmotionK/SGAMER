import math
import pickle
from collections import defaultdict

import torch
import pandas as pd
#from networkx import edges
from torch.nn import functional as F
#from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


dataset_name = 'Amazon_Musical_Instruments'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
    def forward(self,x,edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        return x



if __name__ == '__main__':
    fileFolder = f'../data/{dataset_name}/'
    nodewv = fileFolder + 'nodewv.dic'
    ic_relation = fileFolder + 'item_category.relation'
    ib_relation = fileFolder + 'item_brand.relation'
    ii_relation = fileFolder + 'item_item.relation'
    ui_relation = fileFolder + 'user_item.relation'
    item_brand = pd.read_csv(ib_relation, header=None, sep=',')
    item_category = pd.read_csv(ic_relation, header=None, sep=',')
    item_item = pd.read_csv(ii_relation, header=None, sep=',')
    user_item = pd.read_csv(ui_relation, header=None, sep=',')[[0, 1]]

    x = torch.randn(13305,100).to(device)
    print(x)
    edge_index = item_brand.to_numpy().tolist() + item_category.to_numpy().tolist() + item_item.to_numpy().tolist() + user_item.to_numpy().tolist()

    edge_index = torch.tensor(edge_index).to(device)
    #data = Data(x=x,edge_index=edge_index.t().contiguous(),y=x)


    model = GCN(100,16,100).to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

    print("GCN begin.......")
    for epoch in range(100):
        print(f'epoch：{epoch}')
        out = model(x, edge_index.t().contiguous())
        loss_score = loss(out.to(device),x.to(device)).to(device)

        optimizer.zero_grad()
        loss_score.backward()
        optimizer.step()
    print(out)
    nodewv_dic = defaultdict(torch.Tensor)
    for index,list in enumerate(out):
        nodeid = index
        feature_embedding = [float(x) for x in list]
        #print(feature_embedding)
        nodewv_dic[nodeid] = torch.Tensor(feature_embedding)
    print("finish.....")
    pickle.dump(nodewv_dic, open(nodewv, 'wb'))
    print(len(nodewv_dic))



