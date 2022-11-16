import math

import torch
import pandas as pd
#from networkx import edges
from torch.nn import Parameter, Module, functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

dataset_name = 'Amazon_Musical_Instruments'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x



if __name__ == '__main__':
    fileFolder = f'../data/{dataset_name}/'
    ic_relation = fileFolder + 'item_category.relation'
    ib_relation = fileFolder + 'item_brand.relation'
    ii_relation = fileFolder + 'item_item.relation'
    ui_relation = fileFolder + 'user_item.relation'
    item_brand = pd.read_csv(ib_relation, header=None, sep=',')
    item_category = pd.read_csv(ic_relation, header=None, sep=',')
    item_item = pd.read_csv(ii_relation, header=None, sep=',')
    user_item = pd.read_csv(ui_relation, header=None, sep=',')[[0, 1]]

    x = torch.randn(12834,100)
    edge_index = item_brand.to_numpy().tolist() + item_category.to_numpy().tolist() + item_item.to_numpy().tolist() + user_item.to_numpy().tolist()

    edge_index = torch.tensor(edge_index)
    data = Data(x,edge_index.t().contiguous())
    
    print(data)

    model = GCN(100,16,100).to(device)


