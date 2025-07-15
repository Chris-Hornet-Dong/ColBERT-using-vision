import VisionColBERT
import glob
from PIL import Image
import os

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
    
model = VisionColBERT.VisionColBERT()
  
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