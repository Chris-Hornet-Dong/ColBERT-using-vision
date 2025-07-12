import torch
import torch.nn as nn
import transformers
from transformers import CLIPProcessor, CLIPModel

model=CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
print(torch.cuda.is_available())