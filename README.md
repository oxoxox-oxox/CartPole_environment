# CartPole DQN 训练项目

## 项目介绍

这是一个使用深度 Q 网络 (DQN) 算法训练 CartPole-v1 环境的强化学习项目。项目实现了完整的 DQN 算法，包括经验回放池、目标网络更新等核心组件。

## 项目结构

```
CartPole_environment/
├── env/                # 环境相关代码
│   ├── __pycache__/
│   └── replay_buffer_pool.py  # 经验回放池实现
├── net_model/          # 神经网络模型
│   ├── __pycache__/
│   └── Qnet.py         # Q 网络模型
├── plots/              # 训练结果图表
│   ├── epsilon.png
│   ├── q_values.png
│   ├── rewards.png
│   └── states.png
├── src/                # 算法实现
│   ├── __pycache__/
│   └── DQN.py          # DQN 算法实现
├── README.md           # 项目说明
├── exame_cuda.py       # CUDA 检查脚本
├── main.py             # 主训练脚本
└── plot.py             # 结果可视化脚本
```

## 依赖项

- Python 3.7+
- gymnasium
- numpy
- torch
- matplotlib
- tqdm

## 安装步骤

1. 克隆项目到本地
2. 安装依赖：

   ```bash
   pip install gymnasium numpy torch matplotlib tqdm
   ```

## 使用方法

### 训练模型

运行主脚本开始训练：

```bash
python main.py
```

训练过程中会显示进度条和当前的平均回报。

### 查看训练结果

训练完成后，可以运行绘图脚本查看训练结果：

```bash
python plot.py
```

## 核心组件说明

### 1. Q 网络 (Qnet.py)

一个简单的两层全连接神经网络，用于估计状态-动作价值函数。

### 2. 经验回放池 (replay_buffer_pool.py)

用于存储和采样训练数据，提高训练稳定性和样本效率。

### 3. DQN 算法 (DQN.py)

实现了完整的 DQN 算法，包括：

- ε-贪婪策略选择动作
- 目标网络更新机制
- 损失计算和优化

### 4. 主训练脚本 (main.py)

- 初始化环境和参数
- 执行训练循环
- 收集和存储训练数据
- 更新 Q 网络

## 超参数设置

在 main.py 文件中可以调整以下超参数：

- `lr`: 学习率 (默认: 2e-3)
- `num_episodes`: 训练轮数 (默认: 500)
- `hidden_dim`: 隐藏层维度 (默认: 128)
- `gamma`: 折扣因子 (默认: 0.98)
- `epsilon`: 探索率 (默认: 0.01)
- `target_update`: 目标网络更新频率 (默认: 10)
- `buffer_size`: 经验回放池大小 (默认: 10000)
- `minimal_size`: 开始训练的最小经验数量 (默认: 500)
- `batch_size`: 批量大小 (默认: 64)

## 训练结果

训练完成后，会在 `plots/` 目录下生成以下图表：

- `rewards.png`: 每轮回报曲线
- `epsilon.png`: ε 值变化曲线
- `q_values.png`: Q 值变化曲线
- `states.png`: 状态分布图表

## 技术细节

- 使用 PyTorch 实现神经网络
- 支持 GPU 加速（如果可用）
- 使用 gymnasium 提供的 CartPole-v1 环境
- 实现了目标网络和在线网络的双网络结构
- 使用经验回放池提高训练稳定性

## 扩展建议

- 尝试不同的网络结构和超参数
- 实现更高级的强化学习算法（如 Double DQN、Dueling DQN 等）
- 在其他 gym 环境上测试算法
- 添加模型保存和加载功能

## 许可证

MIT License
