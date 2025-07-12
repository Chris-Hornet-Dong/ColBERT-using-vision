import os

# 创建缓存目录结构
cache_dirs = [
    'D:/huggingface_cache',
    'D:/huggingface_cache/models',
    'D:/huggingface_cache/transformers',
    'D:/huggingface_cache/hub',
    'D:/huggingface_cache/datasets'
]

print("正在创建缓存目录...")
for cache_dir in cache_dirs:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"✅ 创建目录: {cache_dir}")
    else:
        print(f"📁 目录已存在: {cache_dir}")

print("\n缓存目录结构:")
print("D:/huggingface_cache/")
print("├── models/       (模型缓存)")
print("├── transformers/ (Transformers缓存)")
print("├── hub/         (Hub缓存)")
print("└── datasets/    (数据集缓存)")

print("\n目录创建完成！现在可以运行 colbert_example.py 了。") 