#encoding=utf-8


import sys
import os
sys.path.append('..')

import time
import math
import torch
from torchnlp.nn import Attention
import torch.utils.data as Data
from model.util.rank_metrics import ndcg_at_k
from model.util.att import *
from model.util.data_utils import *
import pickle

torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(f'run.py device: {device}')

embedding_size = 100
att_size = embedding_size
latent_size = embedding_size
negative_num = 100
user_n_items = 4 # for each user, it has n items

flag = 'recommendation_drop'
dataset_name = 'Amazon_Musical_Instruments'
#dataset_name = 'Amazon_Automotive'
#dataset_name = 'Amazon_Toys_Games'


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
        #x = self.dropout(x)
        output = F.linear(x, self.weight, self.bias)
        #output = self.dropout(output)
        a, b, c = output.shape
        output = output.reshape((a, c))
        fe = F.log_softmax(output)
        return fe

class AuxiliaryNet(torch.nn.Module):
    def __init__(self,dim):
        super(AuxiliaryNet, self).__init__()
        self.att = Self_Attention_Network(user_item_dim=embedding_size,num_heads=1).to(device)
        self.aux_linear = nn.Linear(embedding_size,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self,input_tensor,is_train=True):
        out,weight = self.att(slf_att_input=input_tensor)
        #out = self.dropout(out)
        out_linear = self.aux_linear(out)
        p_t = self.sigmoid(out_linear)

        if is_train:
            p_t = p_t.repeat(1, 1, 2)
            p_t[:, :, 0] = 1 - p_t[:, :, 0]
            g_hat = F.gumbel_softmax(p_t, 1, hard=False)
            g_t = g_hat[:, :, 1]

        else:
            # size : same as p_t [ batch_size x seq_len x 1]
            m = torch.distributions.bernoulli.Bernoulli(p_t)
            g_t = m.sample()

        one = torch.ones_like(g_t)
        zero = torch.zeros_like(g_t)
        g_t = torch.where(g_t>0.2,one,zero)

        return g_t

class GRU(nn.Module):
    def __init__(self, user_item_dim, input_tensor):
        super(GRU, self).__init__()

        self.auxiliary = AuxiliaryNet(input_tensor.shape[1]).to(device)
        self.instances_slf_att = Self_Attention_Network(user_item_dim=user_item_dim).to(device)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.instances_gru = torch.nn.GRU(input_size=100, hidden_size=100, num_layers=1, batch_first=True).to(device)

    def forward(self, input_tensor):
        
        out, h_state = self.instances_gru(input_tensor.to(device))
        #out = self.dropout(out)
        g_t = self.auxiliary(input_tensor.to(device))
        out, weight = self.instances_slf_att(out.to(device), g_t, is_gate=True)
        
        return out

def item_sequence_attention(input_tensor):
    instances_slf_att = GRU(user_item_dim=latent_size, input_tensor=input_tensor).to(device)
    distance_slf_att = nn.MSELoss()
    optimizer_slf_att = torch.optim.Adam(instances_slf_att.parameters(), lr=0.01, weight_decay=0.00005)
    num_epochs_slf_att = 50
    for epoch in range(num_epochs_slf_att):
        output = instances_slf_att(input_tensor.to(device))
        loss_slf = distance_slf_att(output.to(device), input_tensor.to(device)).to(device)
        optimizer_slf_att.zero_grad()
        loss_slf.backward()
        optimizer_slf_att.step()
    slf_att_embeddings = output.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return slf_att_embeddings

def instances_slf_att(input_tensor):
    instances_slf_att = Self_Attention_Network(user_item_dim=latent_size).to(device)
    distance_slf_att = nn.MSELoss()
    optimizer_slf_att = torch.optim.Adam(instances_slf_att.parameters(), lr=0.01, weight_decay=0.00005)
    num_epochs_slf_att = 50
    for epoch in range(num_epochs_slf_att):
        output,att = instances_slf_att(input_tensor.to(device))
        loss_slf = distance_slf_att(output.to(device), input_tensor.to(device)).to(device)
        optimizer_slf_att.zero_grad()
        loss_slf.backward()
        optimizer_slf_att.step()
    slf_att_embeddings = output.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return slf_att_embeddings

def item_attention(item_input, ii_path):
    """
    here are two item attention.
    the first item attention: ii_path, last_item_att_output
    the second item attention: ii_path, this_item_input
    :param ii_path:
    :param last_item_att_output:
    :param this_item_input:
    :return: item att output
    """
    item_atten = ItemAttention(latent_dim=ii_path.shape[-1], att_size=embedding_size).to(device)
    distance_att = nn.MSELoss()
    optimizer_att = torch.optim.Adam(item_atten.parameters(), lr=0.01, weight_decay=0.00005)
    num_epoch = 10
    for epoch in range(num_epoch):
        output = item_atten(item_input.to(device), ii_path.to(device)).to(device)
        loss_slf = distance_att(output, item_input.to(device))
        optimizer_att.zero_grad()
        loss_slf.backward()
        optimizer_att.step()

    att_embeddings = output.detach().cpu().numpy() # [1,100]
    torch.cuda.empty_cache()
    return att_embeddings

def rec_net(train_loader, test_loader, node_emb, sequence_tensor):
    best_hit_5 = 0.0
    best_hit_10 = 0.0
    best_hit_20 = 0.0
    
    best_ndcg_5 = 0.0
    best_ndcg_10 = 0.0
    best_ndcg_20 = 0.0
    
    all_pos = []
    all_neg = []
    test_data.numpy()
    user_pos = dict()
    user_neg = dict()
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
    recommendation = Recommendation(embedding_size).to(device)
    optimizer = torch.optim.Adam(recommendation.parameters(), lr=0.001)
    for epoch in range(150):
        train_start_time = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch_item_emb = node_emb[batch[:, 1]].reshape((batch.shape[0], 1, embedding_size)).to(device)
            batch_labels = batch[:, 2].to(device)
            batch_sequence_tensor = sequence_tensor[batch[:,0]].reshape((batch.shape[0], 9, embedding_size)).to(device)
            optimizer.zero_grad()
            prediction = recommendation(batch_item_emb, batch_sequence_tensor).to(device)
            loss_train = torch.nn.functional.cross_entropy(prediction, batch_labels).to(device)
            loss_train.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_train.item()
        train_time = time.time() - train_start_time
        print(f'epoch: {epoch}, training loss: {running_loss}, train time: {train_time}')

        if (epoch+1) % 30 != 0:
            continue

        testing_start_time = time.time()

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

        if best_hit_5 < hit_rate_5:
            print('保存hit模型')
            torch.save(recommendation,f'./verify/{dataset_name}/recommendation_hit_{flag}.model')
            best_hit_5 = hit_rate_5
        if best_hit_10 < hit_rate_10:
            best_hit_10 = hit_rate_10
        if best_hit_20 < hit_rate_20:
            best_hit_20 = hit_rate_20
        if best_ndcg_5 < all_ndcg_5:
            print('保存ndcg模型')
            torch.save(recommendation,f'./verify/{dataset_name}/recommendation_ndcg_{flag}.model')
            best_ndcg_5 = all_ndcg_5
        if best_ndcg_10 < all_ndcg_10:
            best_ndcg_10 = all_ndcg_10
        if best_ndcg_20 < all_ndcg_20:
            best_ndcg_20 = all_ndcg_20

        testing_time = time.time() - testing_start_time
        
        print(f"epo:{epoch} | "
              f"HR@5:{hit_rate_5:.4f} | "
              f"HR@10:{hit_rate_10:.4f} | "
              f"HR@20:{hit_rate_20:.4f} | "

              f"NDCG@5:{all_ndcg_5:.4f} | "
              f"NDCG@10:{all_ndcg_10:.4f} | "
              f"NDCG@20:{all_ndcg_20:.4f} | "

              f"best_HR@5:{best_hit_5:.4f} | "
              f"best_HR@10:{best_hit_10:.4f} | "
              f"best_HR@20:{best_hit_20:.4f} | "

              f"best_NDCG@5:{best_ndcg_5:.4f} | "
              f"best_NDCG@10:{best_ndcg_10:.4f} | "
              f"best_NDCG@20:{best_ndcg_20:.4f} | ")

    print('training finish')


if __name__ == '__main__':
#def recommendation_model(dataset_name):
    print('-'*100)
    print(f'{dataset_name}......')
    print('-'*100)

    folder = f'../data/{dataset_name}/'

    # split train and test data
    user_history_file = folder + 'user_history.txt'

    # get train and test links
    N = negative_num  # for each user item, there are N negative samples.
    train_file = folder + 'training_neg_' + str(N) + '.links'
    test_file = folder + 'testing_neg_' + str(N) + '.links'
    train_data, test_data = load_train_test_data(train_file, test_file)

    # load users id and items id
    maptype2id_file = folder + 'map.type2id'
    type2id = pickle.load(open(maptype2id_file, 'rb'))
    users_list = type2id['user']
    user_num = len(users_list)
    items_list = type2id['item']
    item_num = len(items_list)

    # load node embeds
    #node_emb_file = folder + 'node_embedding.dic'
    node_emb_file = folder + 'nodewv.dic'
    node_emb = load_node_tensor(node_emb_file)

    # load ui pairs
    ui_dict = load_ui_seq_relation(user_history_file)

    # load all ui embeddings and ii embeddings
    ui_metapaths_list = ['uibi', 'uibici', 'uici', 'uicibi']
    ii_metapaths_list = ['ibibi', 'ibici', 'ibiui', 'icibi', 'icici', 'iciui', 'iuiui']
    
    user_item_direct_emb_file = folder + 'user_item_dic.wv'
    #user_item_direct_emb = pickle.load(open(user_item_direct_emb_file,'rb'))
    user_item_direct_emb = torch.load(user_item_direct_emb_file)

    item_item_direct_emb_file = folder + 'item_item.wv'
    item_item_direct_emb = load_item_item_wv(item_item_direct_emb_file)
    
    ui_all_paths_emb = load_ui_metapath_instances_emb(ui_metapaths_list, folder, user_num, ui_dict,
                                                      user_item_direct_emb)
    edges_id_dict_file = folder + 'user_history.edges2id'
    edges_id_dict = pickle.load(open(edges_id_dict_file, 'rb'))
    ii_all_paths_emb = load_ii_metapath_instances_emb(folder, user_num, ui_dict, item_item_direct_emb,
                                                      edges_id_dict)
    labels = train_data[:, 2].to(device)
    print(f'labels.shape: {labels.shape}')
    print('loading node embedding, all user-item and item-item paths embedding...finished')

    # 1. user-item instances self attention and for each user item, get one instance embedding.
    print('start training user-item instance self attention module...')
    maxpool = Maxpooling()
    ui_paths_att_emb = defaultdict()
    t = time.time()
    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - t
            print('user ', u, 'time: ', t_here)
        user_item_paths_emb = ui_all_paths_emb[u]
        this_user_ui_paths_att_emb = defaultdict()
        for i in ui_dict[u]:
            if len(ui_all_paths_emb[u][(u, i)]) == 1:
                this_user_ui_paths_att_emb[(u, i)] = ui_all_paths_emb[u][(u, i)]
            else:
                slf_att_input = torch.cuda.FloatTensor(ui_all_paths_emb[u][(u, i)]).unsqueeze(0)
                this_user_ui_paths_att_emb[(u, i)] = instances_slf_att(slf_att_input)
                # user-item instances to one. for each user-item pair, only one instance is needed.
                max_pooling_input = torch.from_numpy(this_user_ui_paths_att_emb[(u, i)])
                get_one_ui = maxpool(max_pooling_input).squeeze(0)
                this_user_ui_paths_att_emb[(u, i)] = get_one_ui
        ui_paths_att_emb[u] = this_user_ui_paths_att_emb
    #ui_batch_paths_att_emb_pkl_file = folder + str(negative_num) + '_ui_batch_paths_att_emb.pkl'
    #pickle.dump(ui_paths_att_emb, open(ui_batch_paths_att_emb_pkl_file, 'wb'))

    # 2. item-item instances slf attention
    print('start training item-item instance self attention module...')
    start_t_ii = time.time()
    ii_paths_att_emb = defaultdict()
    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - start_t_ii
            print('user ', u, 'time: ', t_here)
        item_item_paths_emb = ii_all_paths_emb[u]
        num_item = len(ui_dict[u])
        this_user_ii_paths_att_emb = defaultdict()
        for i_index in range(num_item - 1):
            i1 = ui_dict[u][i_index]
            i2 = ui_dict[u][i_index + 1]
            if len(ii_all_paths_emb[u][(i1, i2)]) == 1:
                this_user_ii_paths_att_emb[(i1, i2)] = ii_all_paths_emb[u][(i1, i2)]
            else:
                slf_att_input = torch.Tensor(ii_all_paths_emb[u][(i1, i2)]).unsqueeze(0)
                this_user_ii_paths_att_emb[(i1, i2)] = instances_slf_att(slf_att_input).squeeze(0)
                this_user_ii_paths_att_emb[(i1, i2)] = torch.from_numpy(this_user_ii_paths_att_emb[(i1, i2)])

        ii_paths_att_emb[u] = this_user_ii_paths_att_emb
    #ii_batch_paths_att_emb_pkl_file =  folder + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    #pickle.dump(ii_paths_att_emb, open(ii_batch_paths_att_emb_pkl_file, 'wb'))

    # 3. user and item embedding
    #ii_batch_paths_att_emb_pkl_file =  folder + str(negative_num) + '_ii_batch_paths_att_emb.pkl'
    #ui_batch_paths_att_emb_pkl_file =  folder + str(negative_num) + '_ui_batch_paths_att_emb.pkl'
    #ii_paths_att_emb = pickle.load(open(ii_batch_paths_att_emb_pkl_file, 'rb'))
    #ui_paths_att_emb = pickle.load(open(ui_batch_paths_att_emb_pkl_file, 'rb'))

    print('start updating user and item embedding...')
    start_t_u_i = time.time()
    sequence_concat = []
    print(f'user_name:{user_num}')
    for u in range(user_num):
        if u % 100 == 0:
            t_here = time.time() - start_t_u_i
            print('user ', u, 'time: ', t_here)
        user_sequence_concat = defaultdict()
        this_user_ui_paths_dic = ui_paths_att_emb[u]
        this_user_ii_paths_dic = ii_paths_att_emb[u]
        
        
        # gru和门控网络
        user_item_sequence = []
        for item_id in ui_dict[u]:
            user_item_sequence.append(node_emb[item_id])
        user_item_sequence = torch.Tensor([item.numpy() for item in user_item_sequence]) # user_item_sequence=torch.Size([12, 100])
        user_item_sequence = user_item_sequence.unsqueeze(0)
        output = item_sequence_attention(user_item_sequence).squeeze() 
        
        
        # for user uid, item1
        u_emb = node_emb[u].reshape((1, -1)).to(device)
        i1_id = ui_dict[u][0]
        u_i1_emb = this_user_ui_paths_dic[(u, i1_id)].reshape((1, -1)).to(device)
        
        #item1_emb = node_emb[i1_id].reshape((1, -1))
        item1_emb = torch.Tensor(output[0].reshape((1, -1)))
        
        # input: u_i1_emb, item1_emb   after attention: the same dimension
        item1_att = item_attention(item1_emb, u_i1_emb.unsqueeze(0)).reshape((1, -1))
        item1_att = torch.from_numpy(item1_att).to(device)

        user_sequence_concat[0] = torch.cat([u_emb, u_i1_emb, item1_att], dim=0).to(device)

        last_item_att = item1_att
        for i_index in range(1, user_n_items):
            i1 = ui_dict[u][i_index - 1]
            i2 = ui_dict[u][i_index]
            item_att_input = this_user_ii_paths_dic[(i1, i2)].unsqueeze(0)
            ii_1 = item_attention(last_item_att, item_att_input).reshape((1, -1))
            ii_1 = torch.from_numpy(ii_1).to(device)
            
            #ii_2 = item_attention(node_emb[i2].unsqueeze(0), item_att_input).reshape((1, -1))
            ii_2 = item_attention(torch.Tensor(output[i_index].reshape((1, -1))), item_att_input).reshape((1, -1))

            ii_2 = torch.from_numpy(ii_2).to(device)
            user_sequence_concat[i_index] = torch.cat([u_emb, ii_1, ii_2], dim=0)
            last_item_att = ii_2
        sequence_concat.append(torch.cat([user_sequence_concat[i] for i in range(0, user_n_items - 1)], 0))
    sequence_tensor = torch.stack(sequence_concat)
    sequence_tensor_pkl_name =  f'./verify/{dataset_name}/sequence_tensor_{flag}.pkl'
    pickle.dump(sequence_tensor, open(sequence_tensor_pkl_name, 'wb'))

    # 4. recommendation
    print('start training recommendation module...')
    #sequence_tensor_pkl_name =  folder + str(negative_num) + '_sequence_tensor.pkl'
    #sequence_tensor = pickle.load(open(sequence_tensor_pkl_name, 'rb'))
    item_emb = node_emb[user_num:(user_num + item_num), :]
    BATCH_SIZE = 128

    train_loader = Data.DataLoader(
        dataset=train_data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  #
        num_workers=5,  #
    )
    test_loader = Data.DataLoader(
        dataset=test_data,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  #
        num_workers=1,  #
    )
    rec_net(train_loader, test_loader, node_emb, sequence_tensor)
