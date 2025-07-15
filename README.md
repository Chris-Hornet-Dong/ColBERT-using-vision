# VisionColBERT
模型结构：
        1. input image——CLIP vision encoder——patch embeddings——linear projection layer——image embeddings
        2. input text——BERT tokenizer——ColBERT encoder——token embeddings
    3. MaxSim