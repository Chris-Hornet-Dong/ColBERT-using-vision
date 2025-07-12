import torch
import transformers
from transformers.pipelines import pipeline

classifier = pipeline("sentiment-analysis")
res = classifier("I've been waiting for a Hugging Face course my whole life.")
print(res)







