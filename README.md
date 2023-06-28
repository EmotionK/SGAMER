# SGAMER
1. If you just want to validate the model on the Amazon Musical Instruments dataset, run the following command:
```
cd model
python recommendation.py
```
---
2. If you want to retrain the model, follow these steps:
## Data Preparation
1. The dataset is available for download at [Amazon](https://nijianmo.github.io/amazon/index.html).，The main download files are "metadata" and "ratings only"
```
mkdir dataset
cd dataset
mkdir {dataset_name} #Different datasets are stored in different folders
```
## Excute
1. Data preprocessing
```
python data_processing.py
```
2. node embedding
```
python embedding_node.py
```
3. Path representation learning for item-item
```
python item_item_representation.py
```
4. Path representation learning for user-item
```
python user_item_representation.py
```
5. Sampling meta-path instances
```
python generate_paths.py
```
6. Meta-path Instance Representation Learning
```
python meta_path_representation.py
```
7. run SGAMER
```
python recommendation.py
```
