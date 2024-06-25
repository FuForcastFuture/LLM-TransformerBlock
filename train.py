"""
Train a model
"""
import os
import torch
import tiktoken
from model import Model
import wandb

# Set GPU max allocation size to 512MB ｜ 将GPU最大分配大小设置为512MB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()  # empty cache if necessary ｜ 如有必要，清空缓存

# Hyperparameters
h_params = {
    "d_model": 1024,  # Define our model dimension architecture ｜ 定义我们的模型维度架构
    "batch_size": 4,  # How many batches per training step ｜ 每个训练步骤有多少批次
    "context_length": 64,  # Length of the token chunk each batch will receive ｜ 每个批次将接收的令牌块的长度
    "num_blocks": 8,  # Number of transformer blocks ｜ Transformer块的数量
    "num_heads": 2,  # Number of heads in Multi-head attention ｜ 多头注意力中的头数
    "dropout": 0.1,  # Dropout rate ｜ 退出率
    "max_iters": 1000,  # Total of training iterations ｜ 总的训练迭代次数
    "learning_rate": 1e-3, # Learning rate ｜ 学习率
    "eval_interval": 50,  # How often to evaluate the model ｜ 评估模型的频率
    "eval_iters": 10,  # Number of iterations to average for evaluation ｜ 用于评估的迭代次数
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if it's available. ｜ 如果可用，使用GPU
    "epochs": 1,
    "TORCH_SEED": 1337
}
torch.manual_seed(h_params["TORCH_SEED"])

# WandB Tracking ｜ WandB跟踪 https://wandb.ai/
# 如需要使用WandB，需要注册账号且拿到access token，
# 并取消注释以下代码：
# run = wandb.init(
#     project="LLMZhang_lesson_2",
#     # Track hyperparameters and run metadata
#     config={
#         "d_model": h_params["d_model"],
#         "batch_size": h_params["batch_size"],
#         "context_length": h_params["context_length"],
#         "max_iters": h_params["max_iters"],
#         "learning_rate": h_params["learning_rate"],
#         "epochs": h_params["epochs"],
#     },
# )


# Prepare Datasets ｜ 准备数据集
with open('data/订单商品名称.csv', 'r', encoding="utf-8") as file:
    text = file.read()

# Using TikToken (Same as GPT3) as tokenizer ｜ 使用TikToken（与GPT3相同）作为分词器
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)
max_token_value = max(tokenized_text)+1  # the maximum value of the tokenized numbers
h_params['max_token_value'] = max_token_value # push max_token_value to hyperparameters for model initialization
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=h_params['device'])

total_tokens = tokenizer.encode_ordinary(text)
print(f"Total: {len(total_tokens):,} tokens")


# Split train and validation data ｜ 分割训练和验证数据
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# Initialize the model ｜ 初始化模型
model = Model(h_params).to(h_params['device'])

# WandB LogMagic ｜ WandB日志魔法
# wandb.watch(model, log_freq=100)

# get batch data ｜ 获取批次数据
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - h_params['context_length'], size=(h_params['batch_size'],))
    x = torch.stack([data[idx:idx + h_params['context_length']] for idx in idxs]).to(h_params['device'])
    y = torch.stack([data[idx + 1:idx + h_params['context_length'] + 1] for idx in idxs]).to(h_params['device'])
    return x, y

# calculate the loss ｜ 计算损失
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(h_params['eval_iters'])
        for k in range(h_params['eval_iters']):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Create the optimizer ｜ 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=h_params['learning_rate'])
for step in range(h_params['max_iters']):
    if step % h_params['eval_interval'] == 0 or step == h_params['max_iters'] - 1:
        losses = estimate_loss()
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:', round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    # Logging Trace ｜ 日志跟踪
    # wandb.log({"train loss": round(losses['train'].item(), 3)})  # WandB validation loss tracking
    # wandb.log({"valid loss": round(losses['valid'].item(), 3)})  # WandB validation loss tracking
    # Backpropagation ｜ 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model ｜ 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'h_params': h_params
}, 'model/model.ckpt')
