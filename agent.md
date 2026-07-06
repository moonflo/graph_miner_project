# AGENT.md

## 1. 项目定位

本项目是一个：

> graph-based latent relation mining system

核心目标：

- 不训练神经网络
- 不做 GPU 训练
- 不做大模型微调
- 不做复杂深度学习 pipeline

## 2. 禁止行为（非常重要）

禁止：

- 引入 PyTorch 训练模型
- 引入 GNN 训练框架（如 DGL training pipeline）
- 自动扩展成多模态系统
- 引入复杂 transformer 训练
- 改成 LLM agent 框架（除非明确要求）

## 3. 允许行为

允许：

- NetworkX 图算法
- sklearn cosine similarity
- embedding API 调用
- OGB 数据集读取
- LLM API 用于解释（仅 optional）
- 可视化（matplotlib / pyvis）

## 4. 核心计算范式

系统必须遵循：

```text
Entity -> Embedding -> Graph Construction -> Graph Algorithms -> Interpretation
```

## 5. 图算法要求

必须实现或支持：

- Adamic-Adar link prediction
- Jaccard coefficient
- Resource allocation（可选）
- Louvain community detection
- shortest path analysis

## 6. 数据要求

支持：

- OGB dataset
- 自定义文本实体数据

OGB 仅用于 evaluation，不用于训练深度模型。

## 7. 设计原则

- simplicity first
- interpretability > complexity
- graph-based reasoning > neural training
