import pickle
import torch
import numpy as np
import torch.utils.data as Data

import sys
sys.path.append('../..')

from model.util.att import *
from model.util.data_utils import *
from torchnlp.nn import Attention
from model.util.rank_metrics import ndcg_at_k
from model.util.data_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'run.py device: {device}')

class Recommendation(nn.Module):
    def __init__(self, in_features):
        """
        :param in_features: mlp input latent: here 100
        :param out_features:  mlp classification number, here neg+1
        """
        super(Recommendation, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(2, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(2))
        self.in_features = in_features
        self.attention1 = Attention(self.in_features)
        self.attention2 = Attention(self.in_features)
        self.dropout = torch.nn.Dropout(p=0.1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, item_emb, sequence_emb):
        """
        :param sequence_emb
        :return:
        """
        x, weights = self.attention1(item_emb, sequence_emb)
        #output = self.dropout(x)
        output = F.linear(x, self.weight, self.bias)
        #output = self.dropout(output)
        a, b, c = output.shape
        output = output.reshape((a, c))
        fe = F.log_softmax(output)
        return fe


recommendation = torch.load('./Amazon_Musical_Instruments_recommendation.model')

folder = '../../data/Amazon_Musical_Instruments/'
sequence_tensor_pkl_name =  folder + '100_sequence_tensor.pkl'
sequence_tensor = pickle.load(open(sequence_tensor_pkl_name, 'rb'))

all_pos = []
all_neg = []
user_pos = dict()
user_neg = dict()
N = 100
embedding_size = 100
train_file = folder + 'training_neg_100.links'
test_file = folder + 'testing_neg_100.links'
train_data,test_data = load_train_test_data(train_file, test_file)
node_emb_file = folder + 'nodewv.dic'
node_emb = load_node_tensor(node_emb_file)

for index in range(test_data.shape[0]):
        user = test_data[index][0].item()
        item = test_data[index][1].item()
        link = test_data[index][2].item()
        if link == 1:
            if user in user_pos.keys():
                user_pos[user].append(item)
            else:
                user_pos[user] = []
                user_pos[user].append(item)
            all_pos.append((index, user, item))
        else:
            if user in user_neg.keys():
                user_neg[user].append(item)
            else:
                user_neg[user] = []
                user_neg[user].append(item)
            all_neg.append((index, user, item))

hit_num_5 = 0
hit_num_10 = 0
hit_num_20 = 0
all_ndcg_5 = 0
all_ndcg_10 = 0
all_ndcg_20 = 0

for i, u_v_p in enumerate(all_pos):
    start = N * i
    end = N * i + N
    p_and_n_seq = all_neg[start:end]
    p_and_n_seq.append(tuple(u_v_p))  # N+1 items

    # 找到embedding，求出score
    scores = []
    for index, userid, itemid in p_and_n_seq:
        # calculate score of user and item
        user_emb = node_emb[userid].reshape((1, 1, embedding_size)).to(device)
        this_item_emb = node_emb[itemid].reshape((1, 1, embedding_size)).to(device)
        this_sequence_tensor = sequence_tensor[userid].reshape((1, 9, embedding_size)).to(device)
        score = recommendation(this_item_emb, this_sequence_tensor)[:, -1].to(device)
        scores.append(score.item())

    normalized_scores = [((u_i_score - min(scores)) / (max(scores) - min(scores))) for u_i_score in scores]
    pos_id = len(scores) - 1
    s = np.array(scores)
    sorted_s = np.argsort(-s)

    if pos_id in sorted_s[0:5]:
        hit_num_5 += 1
        hit_num_10 += 1
        hit_num_20 += 1
    elif pos_id in sorted_s[5:10]:
        hit_num_10 += 1
        hit_num_20 += 1
    elif pos_id in sorted_s[10:20]:
        hit_num_20 += 1
    ndcg_5 = ndcg_at_k(normalized_scores, 5, 0)
    ndcg_10 = ndcg_at_k(normalized_scores, 10, 0)
    ndcg_20 = ndcg_at_k(normalized_scores, 20, 0)
    ndcg_50 = ndcg_at_k(normalized_scores, 50, 0)
    all_ndcg_5 += ndcg_5
    all_ndcg_10 += ndcg_10
    all_ndcg_20 += ndcg_20

all_pos_num = len(all_pos)
hit_rate_5 = hit_num_5 / all_pos_num
hit_rate_10 = hit_num_10 / all_pos_num
hit_rate_20 = hit_num_20 / all_pos_num
all_ndcg_5 = all_ndcg_5 / all_pos_num
all_ndcg_10 = all_ndcg_10 / all_pos_num
all_ndcg_20 = all_ndcg_20 / all_pos_num

print(f"HR@5:{hit_rate_5:.4f} | "
    f"HR@10:{hit_rate_10:.4f} | "
    f"HR@20:{hit_rate_20:.4f} | "
    
    f"NDCG@5:{all_ndcg_5:.4f} | "
    f"NDCG@10:{all_ndcg_10:.4f} | "
    f"NDCG@20:{all_ndcg_20:.4f} | ")
