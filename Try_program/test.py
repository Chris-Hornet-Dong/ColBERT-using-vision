

import transformers
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(bert_tokenizer)


# Load model directly
from transformers import AutoTokenizer, HF_ColBERT, CLIPModel

tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model_colbert = HF_ColBERT.from_pretrained("colbert-ir/colbertv2.0")


model_clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

from datasets import load_dataset

ds = load_dataset("uclanlp/MRAG-Bench")