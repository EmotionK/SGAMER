import json
import pickle
from collections import Counter, defaultdict
from copy import deepcopy
import random as random

import networkx as nx
import torch
import pandas as pd
import numpy as np

from model.embedding_user_item import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #使用设备

user_items_num = 12
dataset_name = '' #数据集名称

# 将数据评分转换为0-1的交互
def binary(ratings,dataset_name):
    if dataset_name=='Amazon_Musical_Instruments': #评分大于等于4的为1，小于4的为0
        y_binary = deepcopy(ratings)
        y_binary[ratings>=4] = 1
        y_binary[ratings<4] = 0
    return y_binary

#获取与用户交互的最近的12个item的整体数据
def get_user_items_num_data(data):

    c_user = Counter(data['userId']) #统计userId出现的数量

    user_12_number = 0
    for key,value in c_user.items():
        if value>=12:
            user_12_number += 1

    print(f'over 12 interaction user number:{user_12_number}')

    most_user = c_user.most_common(user_12_number)  # 5128
    select_users = [i[0] for i in most_user]  # 用户列表

    data_filter_user = data[data['userId'].isin(select_users)]  # 获得选择的用户存在的数据

    c_item = Counter(data_filter_user['itemId'])
    most_item = c_item.most_common()
    select_item = [i[0] for i in most_item]  # 18734

    select_data_flag = data[data['itemId'].isin(select_item) & data['userId'].isin(select_users)]  # (52593, 4)

    re_user = Counter(select_data_flag['userId'])

    select_data = pd.DataFrame() #每个用户最近的12个item
    for userId in re_user:
        user_items = select_data_flag[select_data_flag['userId'] == userId]
        if len(user_items) < user_items_num:
            # select_data = select_data.append(user_items,ignore_index=True)
            continue
        new_user_items = user_items.nlargest(user_items_num, 'timestamp')
        select_data = select_data.append(new_user_items, ignore_index=True)

    # print(len(set(select_data['userId'])))  # 1450
    # print(len(set(select_data['itemId'])))  # 9424
    # print(len(select_data))

    select_data.to_csv(f'./data/{dataset_name}/user_ratings_item.csv', header=None, index=None, sep=',')


    #(n_users,n_items),(X,r,t),(userId_to_index,itemId_to_index),(index_to_userId,index_to_itemId),new_data = reconstruct_data(select_data)##重新构造数据，将userId和itemId转换为从0开始递增

    #new_data.to_csv(f'./data/{dataset_name}/new_data.csv', header=None, index=None, sep=',')  # 将有12个交互的数据保存为csv文件
    #return (n_users,n_items),(X,r,t),(userId_to_index,itemId_to_index),(index_to_userId,index_to_itemId),new_data

#构造用户和项目的交互矩阵
def construct_user_item_interaction_matrix(new_data, user_number, item_number):
    user_item_interaction_matrix = np.zeros([user_number,item_number])#构造n行m列的全零矩阵
    for _,row in new_data.iterrows():
        user_item_interaction_matrix[row[1]][row[0]] = row[2]
    return user_item_interaction_matrix

#为每个用户创建正例和负例
def get_pos_neg_sample_for_user(user_item_interaction_matrix,n):
    pos_samples_idx_mat = np.argwhere(user_item_interaction_matrix > 0).tolist()
    neg_samples_idx_mat = np.argwhere(user_item_interaction_matrix == 0).tolist()

    #负例字典{userId:[itemId,itemId....]}
    neg_samples_idx_dict = {}
    for i in range(n):
        neg_samples_idx_dict[i] = []
    for neg_sample_id in neg_samples_idx_mat:
        neg_samples_idx_dict[neg_sample_id[0]].append(neg_sample_id[1])

    # 正例字典{userId:[itemId,itemId....]}
    pos_samples_idx_dict = {}
    for i in range(n):
        pos_samples_idx_dict[i] = []
    for pos_sample_id in pos_samples_idx_mat:
        pos_samples_idx_dict[pos_sample_id[0]].append(pos_sample_id[1])

    return pos_samples_idx_dict,neg_samples_idx_dict

#get items metas 获取每个项目的category、brand、also_buy
def get_item_meta():
    item_category = []
    item_brand = []
    item_item = []

    with open(f'./dataset/{dataset_name}/meta_{dataset_name}.json','r') as f:
        lines = f.readlines()  # 每一行的列表
        for line in lines:  # 读取一行
            line_json = json.loads(line)  # 将json转化为字典对象
            if 'category' in line_json.keys():
                for cate in line_json['category']:
                    item_category.append([line_json['asin'], 'c_'+cate]) #asin商品唯一标识
            if 'brand' in line_json.keys():
                item_brand.append([line_json['asin'], 'b_'+line_json['brand']])

            if 'also_buy' in line_json.keys():
                for also_item in line_json['also_buy']:
                    item_item.append([line_json['asin'], also_item])

    item_category_df = pd.DataFrame(item_category)
    item_brand_df = pd.DataFrame(item_brand)
    item_item_df = pd.DataFrame(item_item)
    item_category_df.to_csv(f'./data/{dataset_name}/item_category.csv', header=None, index=None, sep=',')
    item_brand_df.to_csv(f'./data/{dataset_name}/item_brand.csv', header=None, index=None, sep=',')
    item_item_df.to_csv(f'./data/{dataset_name}/item_item.csv', header=None, index=None, sep=',')
    print(f'saved items meta to folder: ./data/{dataset_name}')

#根据user_ratings_item.csv提炼item_category.csv,item_brand.csv,item_item.csv
def refine_user_item_category_brand_item():
    user_rate_item_df = pd.read_csv(f'./data/{dataset_name}/user_ratings_item.csv',header=None, sep=',')
    user_set = set(user_rate_item_df[1])
    item_set = set(user_rate_item_df[0])

    #处理item_category
    item_category_df = pd.read_csv(f'./data/{dataset_name}/item_category.csv', header=None, sep=',')
    item_category_df = item_category_df[item_category_df[0].isin(list(item_set))]  # 获取user_rate_item中有的item
    category_set = set(item_category_df[1])  # 获取类别种类

    # 处理item_brand
    item_brand_df = pd.read_csv(f'./data/{dataset_name}/item_brand.csv', header=None, sep=',')
    item_brand_df = item_brand_df[item_brand_df[0].isin(list(item_set))]  # 获取user_rate_item中有的item
    brand_set = set(item_brand_df[1])  # 获取商标种类

    #处理item_item
    item_item_df = pd.read_csv(f'./data/{dataset_name}/item_item.csv', header=None, sep=',')
    item_item_df = item_item_df[(item_item_df[0].isin(list(item_set))) & (item_item_df[1].isin(list(item_set)))]

    #将提炼后的item_category、item_brand、item_item保存为csv文件
    item_category_df.to_csv(f'./data/{dataset_name}/item_category_refine.csv', header=None, index=None, sep=',')
    item_brand_df.to_csv(f'./data/{dataset_name}/item_brand_refine.csv', header=None, index=None, sep=',')
    item_item_df.to_csv(f'./data/{dataset_name}/item_item_refine.csv', header=None, index=None, sep=',')

    name2id = defaultdict(int)  # 实例化字典，value值为int
    id2name = defaultdict(str)
    name2type = defaultdict(str)
    id2type = defaultdict(str)
    type2name = defaultdict(list)
    type2id = defaultdict(list)
    all_nodes = list(user_set) + list(item_set) + list(category_set) + list(brand_set)
    print(f'user_number={len(user_set)}') #1450
    print(f'item_number={len(item_set)}') #9029
    print(f'category_number={len(category_set)}') #540
    print(f'brand_number={len(brand_set)}') #1815
    print(f'all_nodes_number={len(all_nodes)}') #12834
    print(f'all_nodes_set_number={len(set(all_nodes))}') #12834

    i = 0
    #user
    for name in user_set:
        name2id[name] = i #name与id的对应
        id2name[i] = name #id与name的对应
        name2type[name] = 'user'
        id2type[i] = 'user'
        type2name['user'].append(name)
        type2id['user'].append(i)
        i = i + 1

    #item
    for name in item_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'item'
        id2type[i] = 'item'
        type2name['item'].append(name)
        type2id['item'].append(i)
        i = i + 1

    #category
    for name in category_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'category'
        id2type[i] = 'category'
        type2name['category'].append(name)
        type2id['category'].append(i)
        i = i + 1

    #brand
    for name in brand_set:
        name2id[name] = i
        id2name[i] = name
        name2type[name] = 'brand'
        id2type[i] = 'brand'
        type2name['brand'].append(name)
        type2id['brand'].append(i)
        i = i + 1

    refinefolder = f'./data/{dataset_name}/'

    name2idfile = refinefolder + 'map.name2id'
    id2namefile = refinefolder + 'map.id2name'
    name2typefile = refinefolder + 'map.name2type'
    id2typefile = refinefolder + 'map.id2type'
    type2namefile = refinefolder + 'map.type2name'
    type2idfile = refinefolder + 'map.type2id'
    pickle.dump(name2id, open(name2idfile, 'wb'))  # 将对象存储在文件中，序列化
    pickle.dump(id2name, open(id2namefile, 'wb'))
    pickle.dump(name2type, open(name2typefile, 'wb'))
    pickle.dump(id2type, open(id2typefile, 'wb'))
    pickle.dump(type2name, open(type2namefile, 'wb'))
    pickle.dump(type2id, open(type2idfile, 'wb'))

    #使用新的id生成关系文件
    ic_relation = refinefolder + 'item_category.relation'
    ib_relation = refinefolder + 'item_brand.relation'
    ii_relation = refinefolder + 'item_item.relation'
    ui_relation = refinefolder + 'user_item.relation'
    item_category = []
    item_brand = []
    item_item = []
    user_item = []  # user_id, item_id, timestamp
    #item_category
    for _, row in item_category_df.iterrows():
        item_id = name2id[row[0]]
        category_id = name2id[row[1]]
        item_category.append([item_id, category_id])
    item_category_relation = pd.DataFrame(item_category)
    item_category_relation.to_csv(ic_relation, header=None, index=None, sep=',')
    #item_brand
    for _, row in item_brand_df.iterrows():
        item_id = name2id[row[0]]
        brand_id = name2id[row[1]]
        item_brand.append([item_id, brand_id])
    item_brand_relation = pd.DataFrame(item_brand)
    item_brand_relation.to_csv(ib_relation, header=None, index=None, sep=',')
    #item_item
    for _, row in item_item_df.iterrows():
        item1_id = name2id[row[0]]
        item2_id = name2id[row[1]]
        item_item.append([item1_id, item2_id])
    item_item_relation = pd.DataFrame(item_item)
    item_item_relation.to_csv(ii_relation, header=None, index=None, sep=',')
    #user_item
    for _, row in user_rate_item_df.iterrows():
        user_id = name2id[row[1]]
        item_id = name2id[row[0]]
        timestamp = int(row[3])
        user_item.append([user_id, item_id, timestamp])
    user_item_relation = pd.DataFrame(user_item)
    user_item_relation.to_csv(ui_relation, header=None, index=None, sep=',')

    for index, user in enumerate(user_set):
        user_rate_item_df.loc[user_rate_item_df[user_rate_item_df[1] == user].index, 1] = index
    for index, item in enumerate(item_set):
        user_rate_item_df.loc[user_rate_item_df[user_rate_item_df[0] == item].index, 0] = index
    user_rate_item_df.to_csv(f'./data/{dataset_name}/new_data.csv', header=None, sep=',', index=False)
    print(f'generic id finish')

    return len(user_set),len(item_set),user_rate_item_df,len(all_nodes),len(category_set),len(brand_set)

#generate graph
def gen_graph(all_node_number):
    refinefolder = f'./data/{dataset_name}/'
    ic_relation = refinefolder + 'item_category.relation'
    ib_relation = refinefolder + 'item_brand.relation'
    ii_relation = refinefolder + 'item_item.relation'
    ui_relation = refinefolder + 'user_item.relation'
    item_brand = pd.read_csv(ib_relation, header=None, sep=',')
    item_category = pd.read_csv(ic_relation, header=None, sep=',')
    item_item = pd.read_csv(ii_relation, header=None, sep=',')
    user_item = pd.read_csv(ui_relation, header=None, sep=',')[[0, 1]]

    number_nodes = all_node_number  # num_nodes:12834

    G = nx.Graph()
    G.add_nodes_from(list(range(number_nodes)))
    G.add_edges_from(item_brand.to_numpy())
    G.add_edges_from(item_category.to_numpy())
    G.add_edges_from(item_item.to_numpy())
    G.add_edges_from(user_item.to_numpy())
    print(len(G.edges))

    pickle.dump(G, open(f'./data/{dataset_name}/graph.nx', 'wb'))

#user history
def gen_ui_history():
    user_item_relation = pd.read_csv(f'./data/{dataset_name}/user_item.relation', header=None, sep=',')
    users = set(user_item_relation[0])
    user_history_file = f'./data/{dataset_name}/user_history.txt'
    with open(user_history_file, 'w') as f:
        for user in users:
            this_user = user_item_relation[user_item_relation[0] == user].sort_values(2)#user用户与item的交互按照时间戳进行排序
            path = [user] + this_user[1].tolist() #用户与item的列表
            for s in path:
                f.write(str(s) + ' ')
            f.write('\n')

    edges = set()
    with open(user_history_file, 'r') as f:  # 对每个用户的交互项目组成元组
        for line in f.readlines():
            s = line.split()
            uid = s[0]
            node_list = [int(x) for x in s[1:]]  # 交互的项目id列表
            for i in range(len(node_list) - 1):
                if node_list[i] <= node_list[i + 1]:
                    t = (node_list[i], node_list[i + 1])
                else:
                    t = (node_list[i + 1], node_list[i])
                edges.add(t)

    print('edges: ', len(edges))
    edges_id = defaultdict(int)
    id_edges = defaultdict(tuple)
    for i, edge in enumerate(edges):
        edges_id[edge] = i
        id_edges[i] = edge
    edges2id_file = f'./data/{dataset_name}/user_history.edges2id'
    id2edges_file = f'./data/{dataset_name}/user_history.id2edges'
    pickle.dump(edges_id, open(edges2id_file, 'wb'))
    pickle.dump(id_edges, open(id2edges_file, 'wb'))

    edge_path = []  # 所有用户的项目边之间的id

    with open(user_history_file, 'r') as f:
        for line in f.readlines():
            path = []  # 某一个用户的项目交互之间的id
            node_list = [int(x) for x in line.split()[1:]]
            for i in range(len(node_list) - 1):
                if node_list[i] <= node_list[i + 1]:
                    t = (node_list[i], node_list[i + 1])
                else:
                    t = (node_list[i + 1], node_list[i])
                path.append(edges_id[t])
            edge_path.append(path)

    user_history_edge_path_file = f'./data/{dataset_name}/user_history_edge_path.txt'
    with open(user_history_edge_path_file, 'w') as f:
        for path in edge_path:
            # print(len(path), path)
            for s in path:
                # print(s)
                f.write(str(s) + ' ')
            f.write('\n')

#split_train_test
def split_train_test():
    bridge = 2
    train = 4
    test = 6
    testing = []
    training = []
    user_history_file = f'./data/{dataset_name}/user_history.txt'
    with open(user_history_file, 'r') as f:  # 2200
        for line in f:
            s = line.split()
            uid = int(s[0])
            item_history = [int(x) for x in s[1:]]
            if len(item_history) < (bridge + train):#用户交互的项目小于八个
                continue
            else:
                # 划分 train 和 test
                training.append([uid] + item_history[bridge:(bridge + train)])
                testing.append([uid] + item_history[bridge + train:])
    pickle.dump(training, open(f'./data/{dataset_name}/training', 'wb'))
    pickle.dump(testing, open(f'./data/{dataset_name}/testing', 'wb'))

##negative sample
def neg_sample():
    NEGS = [5, 100, 500]
    type2id = pickle.load(open(f'./data/{dataset_name}/map.type2id','rb'))
    all_items = set(type2id['item'])
    training = pickle.load(open(f'./data/{dataset_name}/training', 'rb'))
    testing = pickle.load(open(f'./data/{dataset_name}/testing', 'rb'))
    user_history_dic = defaultdict(list)
    user_history_file = f'./data/{dataset_name}/user_history.txt'
    with open(user_history_file) as f:
        for line in f:
            s = line.split()
            uid = int(s[0])
            user_history_dic[uid] = [int(item) for item in s[1:]]
    for NEG in NEGS:#5,100,500
        training_link = []
        for user_record in training:
            uid = user_record[0]
            positive = [[uid, item, 1] for item in user_record[1:]] #[[208, 2681, 1], [208, 5425, 1], [208, 8486, 1], [208, 1686, 1]]
            #print(f'positive:{positive}')
            bought = set(user_history_dic[uid])
            remain = list(all_items.difference(bought))#difference() 方法用于返回集合的差集，即返回的集合元素包含在第一个集合中，但不包含在第二个集合(方法的参数)中。
            negative = [[uid, item, 0] for item in random.choices(remain, k=len(positive) * NEG)]
            training_link = training_link + positive + negative

        training_link_tf = pd.DataFrame(training_link)
        training_link_tf.to_csv(f'./data/{dataset_name}/training_neg_{str(NEG)}.links', header=None, index=None, sep=',')

        test_link = []
        for user_record in testing:
            uid = user_record[0]
            positive = [[uid, item, 1] for item in user_record[1:]]
            bought = set(user_history_dic[uid])
            remain = list(all_items.difference(bought))
            negative = [[uid, item, 0] for item in random.choices(remain, k=len(positive) * NEG)]
            test_link = test_link + positive + negative

        test_link_tf = pd.DataFrame(test_link)
        test_link_tf.to_csv(f'./data/{dataset_name}/test_neg_{str(NEG)}.links', header=None, index=None, sep=',')
        print(f'save neg {NEG} sampled links ... finish')

def data_processing(datasetName):
    global dataset_name
    dataset_name = datasetName
    print(f'device: {device}')
    data = pd.read_csv(f'./dataset/{dataset_name}/ratings_{dataset_name}.csv', header=None, sep=',',names=['itemId','userId','ratings','timestamp'])
    print(data.shape) #(1512530, 4)

    data['ratings'] = binary(data['ratings'], dataset_name)  # 根据评分判定0 or 1
    data = data[data['ratings'] == 1] #重新定义交互

    get_user_items_num_data(data) #获取与用户交互的最近的12个item的整体数据
    get_item_meta()  # get items metas 获取每个项目的category、brand、also_buy
    user_number,item_number,new_data,all_node_number,category_number,brand_number = refine_user_item_category_brand_item() ##根据user_ratings_item.csv提炼item_category.csv,item_brand.csv,item_item.csv
    gen_graph(all_node_number) #generate graph
    gen_ui_history()#user history
    split_train_test()#split_train_test
    neg_sample()#negative sample

    user_item_interaction_matrix = construct_user_item_interaction_matrix(new_data, user_number, item_number)#构造用户和项目的交互矩阵
    pos_samples_idx_dict, neg_samples_idx_dict = get_pos_neg_sample_for_user(user_item_interaction_matrix, user_number)#为每个用户创建正例和负例

    print('\n -------Data Prep Finished, Starting Training--------')
    return user_number,item_number,pos_samples_idx_dict, neg_samples_idx_dict,category_number,brand_number



















#创建测试字典，并将测试集从数据集中删除
"""def create_test_dict(n,data_new,user_item_interaction_matrix):
    pos_sample_index = np.argwhere(user_item_interaction_matrix==1).tolist() #交互矩阵值为1的索引列表
    pos_user_items_index_dict = {} #用户与项目的交互字典 {user:[item1,item2...]}
    for i in range(n):
        pos_user_items_index_dict[i] = []
    for sample_index in pos_sample_index:
        pos_user_items_index_dict[sample_index[0]].append(sample_index[1])

    df = pd.DataFrame(index=[data_new.userId.to_list(),data_new.itemId.to_list()],columns=['ratings','timestamp'])
    df.index.names=['userId','itemId']
    df.ratings = data_new.ratings.values
    df.timestamp = data_new.timestamp.values

    test_dict = {} #测试集字典，每一个key（userId）对应一个value（itemId）
    ratingIds_to_delete = []
    random.seed(RND_SEED)
    for userId in pos_user_items_index_dict:
        #print(len(pos_user_items_index_dict[userId]))
        itemId_select = random.sample(pos_user_items_index_dict[userId],k=1)#从序列seq中选择n个随机且独立的元素
        test_dict[userId] = itemId_select
        ratingIds_to_delete.append((userId,itemId_select[0]))#需要在原数据中删除的数据
    df.drop(index=ratingIds_to_delete,inplace=True)
    df.reset_index(level=[0,1],inplace=True)

    print("Amount of training samples: ", df.shape[0])
    return df,test_dict"""

#重新构造数据，将userId和itemId转换为从0开始递增
"""def reconstruct_data(data):
    unique_userId = data.userId.unique() #返回userId的唯一值,列表
    userId_to_index = {old_userId:new_userId for new_userId,old_userId in enumerate(unique_userId)}#{1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6,......}
    new_userId = data.userId.map(userId_to_index) #将原数据的userId作为key在user_to_index中查找value

    unique_itemId = data.itemId.unique() #返回itemId的唯一值
    itemId_to_index = {old_itemId:new_itemId for new_itemId,old_itemId in enumerate(unique_itemId)}
    new_itemId = data.itemId.map(itemId_to_index)

    index_to_userId = dict(map(reversed,userId_to_index.items())) #反转字典,key变成value，value变成key
    index_to_itemId = dict(map(reversed,itemId_to_index.items()))

    n_users = unique_userId.shape[0] #1450
    n_items = unique_itemId.shape[0] #9029
    # print(f'n_users={n_users}')
    # print(f'n_items={n_items}')

    X = pd.DataFrame({'userId':new_userId,'itemId':new_itemId}) #原数据中的userId和itemId变成新的 (81516,)
    r = data['ratings'].astype(np.float32)
    t = data['timestamp'].astype(np.float64)
    new_data = pd.DataFrame({'userId':new_userId,'itemId':new_itemId,'ratings':r,'timestamp':data['timestamp']})
    return (n_users,n_items),(X,r,t),(userId_to_index,itemId_to_index),(index_to_userId,index_to_itemId),new_data"""