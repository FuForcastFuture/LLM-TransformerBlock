 
## 环境准备

```bash
pip install -U torch numpy tiktoken
```

可选（wandb用于记录训练过程）：

```bash
pip install wandb
```

wandb (Weights & Bias) 是最通用的模型训练监控的库。它可以帮助你记录模型训练过程中的指标、超参数、模型结构、模型文件等等。你可以在[官网](https://wandb.ai/)上注册一个账号，然后在代码中加入如下代码即可开始记录训练过程：

效果如下图：

![](data/wandb.png)

## 目录结构

- `data/`: 存放用作训练的样本数据集
- `model/`: 存放训练后的模型
- model.py: Transformer模型逻辑代码
- train.py: 训练代码
- inference.py: 推理代码
- finetune.py: 微调代码