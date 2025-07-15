import os
import glob
# 设置环境变量，让模型下载到D盘
os.environ['HF_HOME'] = 'D:/huggingface_cache/models'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache/transformers'
os.environ['HF_HUB_CACHE'] = 'D:/huggingface_cache/hub'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '1000'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip import CLIPProcessor, CLIPModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from PIL import Image



class VisionColBERT(nn.Module):
    
    def __init__(self, 
                 colbert_model_name="colbert-ir/colbertv2.0",
                 clip_model_name="openai/clip-vit-base-patch16",
                 output_dim=256):

        super().__init__()
        
        # 加载ColBERT模型和分词器
        self.colbert = AutoModel.from_pretrained(colbert_model_name)
        self.colbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # ColBERT使用BERT分词器
        
        # 加载CLIP模型和处理器
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 获取维度信息
        colbert_hidden_size = self.colbert.config.hidden_size
        clip_hidden_size = self.clip.vision_model.config.hidden_size
        
        self.clip_projection = nn.Linear(clip_hidden_size, colbert_hidden_size, bias=False)
        
        # 最终输出层：统一到指定维度
        self.output_projection = nn.Linear(colbert_hidden_size, output_dim, bias=False)
        
        print(f"  - ColBERT模型: {colbert_model_name}")
        print(f"  - CLIP模型: {clip_model_name}")
        print(f"  - 输出维度: {output_dim}")

    def encode_text_with_colbert(self, texts):

        # 使用ColBERT的分词器
        inputs = self.colbert_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 通过ColBERT模型
        with torch.no_grad():
            outputs = self.colbert(**inputs)
            # ColBERT使用最后一层的隐藏状态
            text_embeddings = outputs.last_hidden_state
        
        return text_embeddings

    def encode_image_with_clip(self, images):
        # 使用CLIP处理器
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        
        # 通过CLIP视觉模型
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(**inputs)
            patch_embeddings = vision_outputs.last_hidden_state  # (batch, num_patches+1, hidden)
            
            # 去掉CLS token
            patch_embeddings = patch_embeddings[:, 1:, :]  # (batch, num_patches, hidden)
        
        # 投影到ColBERT维度
        patch_embeddings = self.clip_projection(patch_embeddings)
        
        return patch_embeddings

    def compute_colbert_similarity(self, query_embeds, doc_embeds):

        # 计算相似度矩阵
        sim_matrix = torch.matmul(query_embeds, doc_embeds.transpose(-2, -1))
        
        # ColBERT的MaxSim：对每个查询token，找到最相似的文档token
        max_sim_per_query = sim_matrix.max(dim=-1).values  # (batch_size, query_len)
        
        # 对所有查询token的MaxSim求和
        total_similarity = max_sim_per_query.sum(dim=-1)  # (batch_size,)
        
        return total_similarity

    def forward(self, query_texts, doc_images):
        # 使用ColBERT编码文本
        query_embeds = self.encode_text_with_colbert(query_texts)
        
        # 使用CLIP编码图像，然后投影到ColBERT维度
        doc_embeds = self.encode_image_with_clip(doc_images)
        
        # 计算ColBERT相似度
        similarity = self.compute_colbert_similarity(query_embeds, doc_embeds)
        
        return similarity

    def retrieve_images(self, query_text, image_candidates, top_k=5):
        # 使用ColBERT编码查询文本
        query_embeds = self.encode_text_with_colbert([query_text])
        
        # 计算所有候选图像的相似度
        all_scores = []
        for image in image_candidates:
            doc_embeds = self.encode_image_with_clip([image])
            score = self.compute_colbert_similarity(query_embeds, doc_embeds)
            all_scores.append(score.item())
        
        # 排序并返回top-k结果
        sorted_indices = torch.argsort(torch.tensor(all_scores), descending=True)
        top_indices = sorted_indices[:top_k]
        
        top_images = [image_candidates[i] for i in top_indices]
        top_scores = [all_scores[i] for i in top_indices]
        
        return top_images, top_scores

    def get_model_info(self):
        return {
            'colbert_model': self.colbert.config.name_or_path,
            'clip_model': self.clip.config.name_or_path,
            'colbert_hidden_size': self.colbert.config.hidden_size,
            'clip_hidden_size': self.clip.vision_model.config.hidden_size,
            'output_dim': self.output_projection.out_features
        }


path = "try_data"
    
# 查找所有图片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files = []
    
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(path, ext)))
    #image_files.extend(glob.glob(os.path.join(path, ext.upper())))
    
    
print(f"✅ 找到 {len(image_files)} 张图片:")
for i, img_path in enumerate(image_files):
    print(f"  {i+1}. {os.path.basename(img_path)}")
    
model = VisionColBERT()
  
images = []
valid_images = []
    
for img_path in image_files:
    try:
        img = Image.open(img_path)
        # 转换为RGB模式（处理RGBA等其他格式）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
        valid_images.append(img_path)
        print(f"  ✅ {os.path.basename(img_path)} - 尺寸: {img.size}")
    except Exception as e:
        print(f"  ❌ {os.path.basename(img_path)} - 加载失败: {e}")
    
if not images:
    print("❌ 没有成功加载任何图片")    
    
test_queries = [
        "bird",
        "English",
        "Chinese",
        "trees",
        "person"
    ]    
    
# 测试图像检索
for i, query in enumerate(test_queries[:3]):  # 只测试前3个查询
    print(f"\n查询 {i+1}: '{query}'")
    try:
        top_images, top_scores = model.retrieve_images(query, images, top_k=3)
        print(f"  检索结果:")
        for j, (img, score) in enumerate(zip(top_images, top_scores)):
            img_index = images.index(img)
            img_name = os.path.basename(valid_images[img_index])
            print(f"    {j+1}. {img_name} - 相似度: {score:.4f}")
    except Exception as e:
        print(f"  ❌ 检索失败: {e}")
    
# 测试相似度计算
print(f"\n📊 测试相似度计算...")
try:
    # 使用第一个查询和所有图像计算相似度
    query_embeds = model.encode_text_with_colbert([test_queries[0]])
    print(f"\n查询内容：{test_queries[0]}")    
    similarities = []
    for i, img in enumerate(images):
        doc_embeds = model.encode_image_with_clip([img])
        similarity = model.compute_colbert_similarity(query_embeds, doc_embeds)
        similarities.append(similarity.item())
        img_name = os.path.basename(valid_images[i])
        print(f"  {img_name}: {similarity.item():.4f}")
        
    # 找到最相似的图像
    max_idx = similarities.index(max(similarities))
    print(f"\n🏆 最相似的图像: {os.path.basename(valid_images[max_idx])} (相似度: {max(similarities):.4f})")
        
except Exception as e:
    print(f"❌ 相似度计算失败: {e}")
    
# 模型信息
print(f"\n📋 模型信息:")
model_info = model.get_model_info()
for key, value in model_info.items():
    print(f"  {key}: {value}")
    

 