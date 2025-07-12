import os

# 必须在导入 datasets 之前设置环境变量
os.environ['HF_DATASETS_CACHE'] = 'D:/my_datasets_cache'

# 然后再导入和加载数据集
from datasets import load_dataset

print("正在加载数据集...")
ds = load_dataset("uclanlp/MRAG-Bench")

print("\n" + "="*60)
print("数据集基本信息")
print("="*60)

# 1. 基本结构
print(f"数据集类型: {type(ds)}")
print(f"数据集分割: {list(ds.keys())}")

# 2. 查看每个分割的信息
for split_name, split_data in ds.items():
    print(f"\n--- {split_name} 分割 ---")
    print(f"样本数量: {len(split_data)}")
    print(f"特征: {split_data.features}")
    
    # 显示前几个样本的结构
    if len(split_data) > 0:
        print(f"第一个样本的键: {list(split_data[0].keys())}")
        print(f"第一个样本内容:")
        for key, value in split_data[0].items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")

print("\n" + "="*60)
print("数据集统计信息")
print("="*60)

# 3. 统计信息
total_samples = sum(len(split) for split in ds.values())
print(f"总样本数: {total_samples}")

for split_name, split_data in ds.items():
    print(f"{split_name}: {len(split_data)} 样本")

print("\n" + "="*60)
print("数据集详细信息")
print("="*60)

# 4. 数据集信息
print(f"数据集信息: {ds.info}")

# 5. 查看数据集描述
if hasattr(ds, 'description'):
    print(f"数据集描述: {ds.description}")

print("\n" + "="*60)
print("示例数据查看")
print("="*60)

# 6. 查看具体样本
if 'train' in ds:
    print("训练集前3个样本:")
    for i in range(min(3, len(ds['train']))):
        print(f"\n样本 {i+1}:")
        sample = ds['train'][i]
        for key, value in sample.items():
            if isinstance(value, str):
                # 截断长文本
                display_value = value[:200] + "..." if len(value) > 200 else value
                print(f"  {key}: {display_value}")
            else:
                print(f"  {key}: {value}")

if 'test' in ds:
    print(f"\n测试集前2个样本:")
    for i in range(min(2, len(ds['test']))):
        print(f"\n样本 {i+1}:")
        sample = ds['test'][i]
        for key, value in sample.items():
            if isinstance(value, str):
                display_value = value[:200] + "..." if len(value) > 200 else value
                print(f"  {key}: {display_value}")
            else:
                print(f"  {key}: {value}")

print("\n" + "="*60)
print("数据集操作示例")
print("="*60)

# 7. 数据集操作示例
print("数据集支持的操作:")
print("- ds['train'] - 访问训练集")
print("- ds['test'] - 访问测试集")
print("- ds['validation'] - 访问验证集（如果存在）")
print("- len(ds['train']) - 获取训练集大小")
print("- ds['train'][0] - 获取第一个样本")
print("- ds['train'].features - 获取特征信息")
print("- ds['train'].column_names - 获取列名")

# 8. 列名信息
if 'train' in ds:
    print(f"\n训练集列名: {ds['train'].column_names}")
    print(f"特征详情:")
    for feature_name, feature_type in ds['train'].features.items():
        print(f"  {feature_name}: {feature_type}") 