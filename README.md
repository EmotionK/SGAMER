# SGAMER
1. 如果你只是想验证该模型在数据集**Amazon Musical Instruments**上的结果，只需要运行以下的命令
```
>>cd model
>>python recommendation.py
```
3. 如果你想重新训练该模型，可以按照以下步骤执行：
数据准备
1. 相关数据集可以在Amazon中下载，主要下载"metadata"和"ratings only"两个文件
mkdir dataset
cd dataset
mkdir {dataset_name} #不同数据集存放在不同的文件夹中
Excute
1. 数据预处理
python data_processing.py
2. 将节点进行嵌入处理
python embedding_node.py
3. user-item的路径表示学习
python item_item_representation.py
4. item-item的路径表示学习
python user_item_representation.py
5. 元路径实例采样
python generate_paths.py
6. 元路径实例表示学习
python meta_path_representation.py
7. run SGAMER
python recommendation.py
