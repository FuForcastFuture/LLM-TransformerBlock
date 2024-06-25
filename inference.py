# -*- coding: utf-8 -*-
"""
Sample from a trained model ｜ 从训练好的模型中生成样本
"""
import os
import torch
import tiktoken
from model import Model

# Load the model and hyperparameters ｜ 加载模型和超参数
checkpoint = torch.load('model/model.ckpt')
h_params = checkpoint['h_params']
model = Model(h_params)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(h_params['device'])

# Load the Tiktoken tokenizer ｜ 加载Tiktoken分词器
encoding = tiktoken.get_encoding("cl100k_base")

start = "农夫山泉 "
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=h_params['device'])[None, ...])

# run generation ｜ 运行生成
with torch.no_grad():
    y = model.generate(x, max_new_tokens=200, temperature=1.0, top_k=None)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')

# Optionally, print model total of parameters ｜ 可选地，打印模型参数总数
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model param size: {total_params:,}")


