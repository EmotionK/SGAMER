import math

import torch
import pandas as pd
#from networkx import edges
from torch.nn import Parameter, Module, functional as F

dataset_name = 'Amazon_Musical_Instruments'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
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



    model = GCN(100,16,100).to(device)


