import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# -----------------------------------------------------------------------------
# 第 1 步：加载数据集
# -----------------------------------------------------------------------------
# 加载 IMDB 电影评论数据集，它包含 'train' 和 'test' 两部分。
# 我们只取一小部分样本来加速演示过程。
dataset = load_dataset("imdb")
# 为了快速演示，可以只取一小部分数据
# train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
# test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))


# -----------------------------------------------------------------------------
# 第 2 步：加载分词器并预处理数据
# -----------------------------------------------------------------------------
# 加载与我们要微调的模型相匹配的分词器
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 创建一个预处理函数，它会将文本转换为模型可以理解的数字 ID
def preprocess_function(examples):
    # padding="max_length" 会将所有样本补齐到相同长度
    # truncation=True 会截断超过模型最大长度的样本
    return tokenizer(examples["text"], truncation=True)

# 使用 .map() 方法将预处理函数应用到整个数据集上
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# -----------------------------------------------------------------------------
# 第 3 步：加载预训练模型
# -----------------------------------------------------------------------------
# 加载带有序列分类头的 DistilBERT 模型。
# num_labels=2 告诉模型我们要做的是一个二分类任务 (正面/负面)。
# 这个类会自动在 DistilBERT 基础模型上添加一个分类头。
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# -----------------------------------------------------------------------------
# 第 4 步：定义评估指标
# -----------------------------------------------------------------------------
# 加载准确率评估指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 获取概率最高的预测结果
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# -----------------------------------------------------------------------------
# 第 5 步：定义训练参数
# -----------------------------------------------------------------------------
# TrainingArguments 是一个包含了所有训练超参数的类。
training_args = TrainingArguments(
    output_dir="my_awesome_model",      # 训练结果的输出目录
    learning_rate=2e-5,                 # 学习率
    per_device_train_batch_size=16,     # 训练时的 batch size
    per_device_eval_batch_size=16,      # 评估时的 batch size
    num_train_epochs=2,                 # 训练的轮数
    weight_decay=0.01,                  # 权重衰减
    evaluation_strategy="epoch",        # 每个 epoch 结束后进行一次评估
    save_strategy="epoch",              # 每个 epoch 结束后保存一次模型
    load_best_model_at_end=True,        # 训练结束后加载最佳模型
)

# -----------------------------------------------------------------------------
# 第 6 步：创建并启动 Trainer
# -----------------------------------------------------------------------------
# Trainer 类将所有组件（模型、训练参数、数据集、评估函数）整合在一起
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("gpu is available: ", torch.cuda.is_available())
# 开始训练！
print("开始模型微调...")
trainer.train()
print("微调完成！")

# -----------------------------------------------------------------------------
# 第 7 步：使用微调好的模型
# -----------------------------------------------------------------------------
# 训练完成后，模型和分词器会保存在 output_dir 中
# 我们可以直接用 pipeline 来加载这个本地模型进行推理
from transformers import pipeline

# "my_awesome_model" 是你上面定义的 output_dir
sentiment_pipeline = pipeline("sentiment-analysis", model="my_awesome_model")

print("\n使用微调后的模型进行预测:")
print(sentiment_pipeline("This new movie is fantastic! I loved it."))
print(sentiment_pipeline("The plot was predictable and the acting was mediocre."))