import os

from data_processing import data_processing
#from model.embedding_user_item import embedding_user_item
#from model.embedding_node import embedding_node
#from model.generate_paths import gen_instances
#from model.item_item_representation import item_item_repersentation
#from model.meta_path_instances_representation import meta_path_instances_representation
#from model.recommendation_model import recommendation_model
#from model.user_history import user_history
#from model.user_item_representation import user_item_representation

base_path = os.getcwd() + '/' #获取当前路径

dataset_name = 'Amazon_Musical_Instruments'
#dataset_name = 'Amazon_Automotive'
#dataset_name ='Amazon_Toys_Games'

print(f'{dataset_name} data processing.....')
user_number,item_number,category_number,brand_number = data_processing(dataset_name)

#print(f'{dataset_name} embedding_user_item.....')
#embedding_user_item(dataset_name,user_number,item_number)

print(f'{dataset_name} embedding_category_brand.....')
#embedding_node(dataset_name,user_number,item_number)

print(f'{dataset_name} item_item_repersentation.....')
#item_item_repersentation(dataset_name)

print(f'{dataset_name} user_item_representation.....')
#user_item_representation(dataset_name)

print(f'{dataset_name} gen_instances.....')
#gen_instances(dataset_name,user_number,item_number,category_number,brand_number)

print(f'{dataset_name} get meta_path_instances_representation.....')
#meta_path_instances_representation(dataset_name)

print(f'{dataset_name} get sequence item-item paths for each user.....')
#user_history(dataset_name)

print(f'{dataset_name} train recommendation_model.....')
#recommendation_model(dataset_name)
