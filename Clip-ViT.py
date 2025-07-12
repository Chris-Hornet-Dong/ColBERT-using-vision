
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import PIL
class VisionColBERT(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16", output_dim=256):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        # 可选：降维到ColBERT的维度
        self.linear = nn.Linear(self.clip.vision_model.config.hidden_size, output_dim, bias=False)#降维的线性层

    def encode_image(self, images):
        # images: PIL images or tensor, batch of images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        vision_outputs = self.clip.vision_model(**inputs)
        patch_embeddings = vision_outputs.last_hidden_state  # (batch, num_patches+1, hidden)
        patch_embeddings = patch_embeddings[:, 1:, :]  # 去掉CLS token
        patch_embeddings = self.linear(patch_embeddings)  # (batch, num_patches, output_dim)
        patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        return patch_embeddings

    def forward(self, query_embeds, doc_images):
        # query_embeds: (batch, query_len, output_dim)
        # doc_images: list of PIL images or tensor
        doc_embeds = self.encode_image(doc_images)  # (batch, doc_len, output_dim)
        # 计算MaxSim
        # 假设batch=1，简化演示
        query_embeds = query_embeds[0]  # (query_len, output_dim)
        doc_embeds = doc_embeds[0]      # (doc_len, output_dim)
        # (query_len, doc_len)
        sim_matrix = torch.matmul(query_embeds, doc_embeds.t())
        maxsim = sim_matrix.max(dim=1).values.sum()
        return maxsim

# 用法示例
model = VisionColBERT()
images = [PIL.Image.open("test_image.jpg")]
doc_embeds = model.encode_image(images)
print(doc_embeds)