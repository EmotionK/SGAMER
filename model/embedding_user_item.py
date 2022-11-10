import random
import pickle
import torch
import math
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.distributions import Categorical
import torch.utils.data as Data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用设备
RND_SEED = 123
parameters_arr = [3,64,0.5,0.005,0.005] #3个角色，64维的潜在向量，alpha=0.5，lambda_p=lambda_n=0.005
num_epochs = 200 #训练次数
batch_size = 256
lr = 0.001 #learning rate

class ReviewsIterator:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)#转为数组类型

        if shuffle:#将X和y随机排列
            index = np.random.permutation(X.shape[0])#随机排列
            X, y = X[index], y[index]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))#向上取整
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]

class CustomLoss(torch.nn.Module):

    def __init__(self, lamda_pos=0.005, lamda_neg=0.005, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.lamda_pos = lamda_pos
        self.lamda_neg = lamda_neg
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, model, users, items, prediction_input, labels, personas_scores_input):
        # The loss function without the entropy term over the personas vectors
        if personas_scores_input is None:
            CE_loss = self.loss_function(prediction_input, labels)  # 计算交叉熵损失
            return CE_loss, CE_loss, torch.Tensor([0]), torch.Tensor([0])

        # The loss function for the AMP-CF model
        CE_loss = self.loss_function(prediction_input, labels)
        personas_scores = personas_scores_input.clone()
        entropy_pos = torch.sum(Categorical(probs=personas_scores[:, :1, :]).entropy())#Categorical:以值为概率返回索引
        entropy_negs = torch.sum(torch.sum(Categorical(probs=personas_scores[:, 1:, :]).entropy(), dim=-1) \
                                 / (personas_scores.shape[1] - 1))
        total_loss = (self.alpha) * (CE_loss) + (1 - self.alpha) * (
                    self.lamda_pos * entropy_pos - self.lamda_neg * entropy_negs)
        return (total_loss, CE_loss, entropy_pos, entropy_negs)

class Collaborative_Filtering(torch.nn.Module):

    #### An object that contains all the models in the article (both initialization and forward) ####

    def __init__(self, n_users, n_items, n_factors, n_personas):
        super().__init__()

        self.n_personas = n_personas
        self.device = device

        # AMPCF
        self.emb_dimension = n_factors
        '''
             torch.nn.Parameter继承torch.Tensor，其作用将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，
            并将这个参数绑定到module里面，成为module中可训练的参数。
        '''
        self.user_factors = torch.nn.Parameter(torch.randn(n_users, n_personas, n_factors)) # 1450,3,64
        self.item_factors = torch.nn.Parameter(torch.randn(n_items, n_factors)) # 9029, 64

         #xavier初始化方法中服从正态分布N(mean=0, std)
        torch.nn.init.xavier_normal_(self.user_factors)
        torch.nn.init.xavier_normal_(self.item_factors)

    def forward(self, user, items):
        u = self.user_factors[user].squeeze()  # 256,3,64 torch.squeeze：将tensor中大小为1的维度删除
        v = self.item_factors[items]  # 256,5,64
        r = torch.einsum('bsf,bpf->bsp', [v, u]) # 256,4,
        attentive_scores = F.softmax(r, dim=-1)
        attentive_user = torch.einsum('bpf,bsp->bsf', [u, attentive_scores])
        pred = torch.einsum('bsf,bsf->bs', [attentive_user, v])
        return (pred, attentive_scores)

'''This object contains the trained model, the variables and the functions as explained in the comments.
       Most variables are initialized during the run or set at the end.
       The user has the option to change the following fields in the object:
       num_neg_samples, num_random_samples, k.'''
class Model:
    def __init__(self, n_factors, n_personas,n, m,
                 datasets,pos_samples_idx_dict,neg_samples_idx_dict, dataset_sizes,num_epochs, batch_size, lr, alpha, lamda_pos, lamda_neg):

        self.device = device
        self.n = n  # num of users
        self.m = m  # num of items
        self.n_factors = n_factors  # number of latent dimensions
        self.n_personas = n_personas  # number of personas per user or model indicator
        self.batch_size = batch_size  # batch size
        self.n_epochs = num_epochs  # number of epochs
        self.lr = lr  # learning rate
        self.wd = 0  # weight decay in Adam optimize
        self.alpha = alpha  # a hyperparameter in the loss function
        # self.lamda_pos = lamda_pos  # a hyperparameter in the loss function
        # self.lamda_neg = lamda_neg  # a hyperparameter in the loss function
        self.num_neg_samples = 4  # number of negative items per positive item
        self.num_samples = self.num_neg_samples + 1  # negative samples + positive sample
        self.num_random_samples = 99  # number of random items in test (the 100th item is the test item)
        self.losses = []  # loss array
        self.target = torch.LongTensor([0] * self.batch_size).to(self.device)  # loss function labels
        self.loss_func = CustomLoss(alpha=alpha, lamda_pos=lamda_pos, lamda_neg=lamda_neg)  # the loss function
        # self.userId_to_index = userId_to_index  # user ID to index
        # self.index_to_userId = index_to_userId  # user index to ID
        # self.movieId_to_index = movieId_to_index  # item ID to inde
        # self.index_to_movieId = index_to_movieId  # item index to ID
        self.datasets = datasets  # the dataset of the model after data processing
        self.dataset_sizes = dataset_sizes  # number of train samples
        self.top_movies_idxs_everyone = None  # list of top 30 movies (indices) for each user
        self.batches_dict = {}  # batches dictionary
        self.pos_samples_idx_dict = pos_samples_idx_dict  # positive samples for each user
        self.neg_samples_idx_dict = neg_samples_idx_dict  # negative samples for each user
        # self.test_dict = test_dict  # dictionary of users and their test items
        #self.test_hashtable = self.get_test_hashtable()  # the test items for each user
        self.model = Collaborative_Filtering(n, m, n_factors, n_personas).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd,
                                          amsgrad=False)  # the optimizer

    def train_model(self,dataset_name):

        self.model.train()
        train_data = torch.LongTensor(self.datasets.to_numpy())
        train_loader = Data.DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True,drop_last=True)

        for epoch in range(self.n_epochs): #训练200轮

            print("Epoch: {}".format(epoch + 1), end='\n', flush=True)
            stats = {'epoch': epoch + 1, 'total': self.n_epochs}

            phase = 'train'
            training = True
            running_loss_ce = 0.0
            running_loss_ent_pos = 0.0
            running_loss_ent_neg = 0.0
            running_loss = 0.0
            n_batches = 0
            i = 0

            for step,batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(training):  # compute gradients only during 'train' phase
                    batch = batch.numpy()
                    batch_user_idxs = batch.T[1] #随机采样的256个userId列表
                    batch_item_idxs = []
                    for user_idx in batch_user_idxs:  # 为每一个batch中的用户采样正样本1个，负样本4个
                        neg_items_idxs = random.sample(self.neg_samples_idx_dict[user_idx.item()],
                                                        k=self.num_neg_samples)
                        pos_item_idx = random.sample(self.pos_samples_idx_dict[user_idx.item()], k=1)
                        item_idxs = pos_item_idx + neg_items_idxs #为每个用户选择的item
                        batch_item_idxs.append(item_idxs)
                    users = torch.LongTensor([batch_user_idxs]).to(self.device)#user的tensor
                    items = torch.LongTensor(batch_item_idxs).to(self.device)  # 包含正负样本的item的tensor
                    predictions, personas_scores = self.model(users.t(), items)
                    loss, CE_loss, entropy_pos, entropy_negs = self.loss_func(self.model,
                                                                              self.model.user_factors[users].reshape(
                                                                                  -1),
                                                                              self.model.item_factors[items].reshape(
                                                                                  -1), predictions, self.target,
                                                                              personas_scores)
                    loss.backward()
                    self.optimizer.step()
                    del users
                    del items
                    del predictions
                    del personas_scores
                    torch.cuda.empty_cache()
                running_loss += loss.item()
                running_loss_ce += CE_loss.item()
                running_loss_ent_pos += entropy_pos.item()
                running_loss_ent_neg += entropy_negs.item()
                self.losses.append(loss.item() / self.batch_size)
                epoch_loss = running_loss / self.dataset_sizes
                epoch_loss_ce = running_loss_ce / self.dataset_sizes
                epoch_loss_pos = running_loss_ent_pos / self.dataset_sizes
                epoch_loss_neg = running_loss_ent_neg / self.dataset_sizes

                stats[phase] = epoch_loss
            print("CE Loss: {:.4f} Entr Pos Loss: {:.4f} Entr Neg Loss: {:.4f}\n".format(epoch_loss_ce, epoch_loss_pos, epoch_loss_neg), end='', flush=True)
        pickle.dump(self.model.user_factors, open(f'./data/{dataset_name}/torch.user_embedding', 'wb'))
        pickle.dump(self.model.item_factors, open(f'./data/{dataset_name}/torch.item_embedding', 'wb'))

def embedding_user_item(dataset_name,n,m,pos_samples_idx_dict, neg_samples_idx_dict,):
    num_personas, latent_dimension, alpha, lambda_p, lambda_n = parameters_arr
    rnd_seed = RND_SEED
    random.seed(RND_SEED)
    torch.manual_seed(RND_SEED)
    np.random.seed(RND_SEED)
    data_new = pd.read_csv(f'./data/{dataset_name}/new_data.csv', header=None, sep=',')
    model = Model(latent_dimension, num_personas, n, m, data_new, pos_samples_idx_dict, neg_samples_idx_dict,
                  len(data_new), num_epochs=num_epochs, batch_size=batch_size, lr=lr,
                  alpha=alpha, lamda_pos=lambda_p, lamda_neg=lambda_n)
    model.train_model(dataset_name)