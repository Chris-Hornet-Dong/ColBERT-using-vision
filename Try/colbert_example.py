import torch
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 第 1 步：加载 ColBERT 模型和分词器
# -----------------------------------------------------------------------------
print("正在加载 ColBERT 模型...")

# 注意：ColBERT 使用标准的 BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")

# 将模型设置为评估模式
model.eval()

# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"模型已加载到设备: {device}")

# -----------------------------------------------------------------------------
# 第 2 步：准备文档集合（语料库）
# -----------------------------------------------------------------------------
documents = [
    "ColBERT is a neural retrieval model that uses late interaction.",
    "BERT is a transformer-based model for natural language processing.",
    "Information retrieval is the process of finding relevant documents.",
    "Machine learning algorithms can improve search results.",
    "Deep learning models have revolutionized NLP tasks."
]

print(f"\n文档集合包含 {len(documents)} 个文档:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc}")

# -----------------------------------------------------------------------------
# 第 3 步：对文档进行编码
# -----------------------------------------------------------------------------
def encode_documents(docs, tokenizer, model, device):
    """对文档集合进行编码"""
    encoded_docs = []
    
    for doc in docs:
        # 对文档进行分词
        inputs = tokenizer(
            doc,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # 获取文档的嵌入表示
        with torch.no_grad():
            outputs = model(**inputs)
            # ColBERT 使用最后一层的隐藏状态
            doc_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
        
        encoded_docs.append(doc_embeddings)
    
    return encoded_docs

print("\n正在编码文档...")
encoded_documents = encode_documents(documents, tokenizer, model, device)
print("文档编码完成！")

# -----------------------------------------------------------------------------
# 第 4 步：对查询进行编码
# -----------------------------------------------------------------------------
def encode_query(query, tokenizer, model, device):
    """对查询进行编码"""
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        query_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
    
    return query_embeddings

# -----------------------------------------------------------------------------
# 第 5 步：计算相似度并检索
# -----------------------------------------------------------------------------
def compute_similarity(query_emb, doc_emb):
    """计算查询和文档之间的相似度"""
    # 使用最大池化来获得文档级别的表示
    doc_emb_pooled = torch.mean(doc_emb, dim=0).unsqueeze(0)  # [1, hidden_dim]
    query_emb_pooled = torch.mean(query_emb, dim=0).unsqueeze(0)  # [1, hidden_dim]
    
    # 计算余弦相似度
    similarity = torch.cosine_similarity(query_emb_pooled, doc_emb_pooled, dim=1)
    return similarity.item()

def retrieve_documents(query, documents, encoded_docs, tokenizer, model, device, top_k=3):
    """检索最相关的文档"""
    print(f"\n查询: '{query}'")
    
    # 编码查询
    query_emb = encode_query(query, tokenizer, model, device)
    
    # 计算与所有文档的相似度
    similarities = []
    for i, doc_emb in enumerate(encoded_docs):
        sim = compute_similarity(query_emb, doc_emb)
        similarities.append((i, sim))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 返回 top-k 结果
    print(f"\nTop-{top_k} 检索结果:")
    for rank, (doc_idx, score) in enumerate(similarities[:top_k]):
        print(f"{rank+1}. 文档 {doc_idx+1}: {documents[doc_idx]}")
        print(f"   相似度分数: {score:.4f}")
    
    return similarities[:top_k]

# -----------------------------------------------------------------------------
# 第 6 步：测试检索功能
# -----------------------------------------------------------------------------
test_queries = [
    "What is ColBERT?",
    "How does machine learning work?",
    "Tell me about information retrieval",
    "What are transformer models?"
]

print("\n" + "="*60)
print("开始文档检索测试")
print("="*60)

for query in test_queries:
    retrieve_documents(query, documents, encoded_documents, tokenizer, model, device)
    print("-" * 40)

print("\n检索测试完成！") 