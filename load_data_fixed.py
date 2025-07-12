import os

# 必须在导入 datasets 之前设置环境变量
os.environ['HF_DATASETS_CACHE'] = 'D:/my_datasets_cache'

# 然后再导入和加载数据集
from datasets import load_dataset

ds = load_dataset("uclanlp/MRAG-Bench")
print(ds) 
print(help(ds))