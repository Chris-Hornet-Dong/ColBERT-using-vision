import colbert_modules

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
    colbert_modules.retrieve_documents(query, documents, encoded_documents, tokenizer, model, device)
    print("-" * 40)

print("\n检索测试完成！") 