# PyTorch SERL 实现

这是 HIL-SERL 项目的 PyTorch 简化实现，专注于三个核心训练过程：

1. **奖励分类器训练**
2. **纯离线强化学习训练**  
3. **混合模式训练（在线+离线）**

## 主要特性

- 🔥 **纯 PyTorch 实现**：摆脱 JAX 依赖，使用广泛支持的 PyTorch
- 🚀 **设备兼容**：自动支持 CUDA、MPS(Apple Silicon) 和 CPU
- 📦 **简化架构**：保留核心功能，去除复杂配置
- 🎯 **专注核心**：只实现最重要的三个训练流程

## 目录结构

```
pytorch_serl/
├── agents/           # SAC智能体实现
│   └── sac.py       # 标准SAC和混合SAC智能体
├── networks/         # 神经网络架构
│   ├── resnet.py    # ResNet编码器
│   ├── mlp.py       # MLP网络
│   ├── actor_critic.py  # Actor-Critic网络
│   └── classifier.py    # 奖励分类器
├── data/            # 数据处理
│   └── replay_buffer.py # 重放缓冲区
├── utils/           # 工具函数
│   └── device.py    # 设备配置
├── train_classifier.py  # 分类器训练脚本
├── train_offline.py     # 离线训练脚本
└── train_hybrid.py      # 混合训练脚本
```

## 快速开始

### 1. 环境要求

```bash
pip install torch torchvision tqdm numpy
```

### 2. 训练奖励分类器

```bash
cd pytorch_serl

# 训练二分类奖励分类器
python train_classifier.py \
    --success_dir ./data/success \
    --failure_dir ./data/failure \
    --image_keys image \
    --output_dir ./classifier_ckpt \
    --batch_size 64 \
    --num_epochs 150
```

### 3. 纯离线训练

```bash
# 使用演示数据进行离线强化学习
python train_offline.py \
    --demo_paths ./data/demos.pkl \
    --image_keys image \
    --setup_mode single-arm-fixed-gripper \
    --batch_size 256 \
    --max_steps 50000 \
    --save_dir ./offline_ckpt
```

### 4. 混合模式训练

```bash
# 同时使用离线数据和在线交互
python train_hybrid.py \
    --demo_paths ./data/demos.pkl \
    --image_keys image \
    --setup_mode single-arm-learned-gripper \
    --max_steps 100000 \
    --batch_size 256 \
    --save_dir ./hybrid_ckpt
```

## 支持的模式

### 设置模式 (setup_mode)

- `single-arm-fixed-gripper`: 单臂固定抓取器（预抓取）
- `single-arm-learned-gripper`: 单臂学习抓取器  
- `dual-arm-fixed-gripper`: 双臂固定抓取器
- `dual-arm-learned-gripper`: 双臂学习抓取器

### 网络架构

- **编码器**: ResNet-10 (基于ResNet-18实现)
- **策略网络**: MLP + Tanh分布
- **评论家网络**: 集成Q网络 (默认2个)
- **抓取评论家**: DQN风格Q网络 (用于混合策略)

## 数据格式

### 演示数据格式
```python
transition = {
    'observations': {
        'image': np.array,  # 图像观测 (H, W, C)
        'state': np.array   # 状态信息(可选)
    },
    'actions': np.array,    # 动作 (连续动作 + 抓取动作)
    'next_observations': {  # 下一步观测
        'image': np.array,
        'state': np.array
    },
    'rewards': float,       # 奖励
    'dones': bool,         # 是否结束
    'masks': float,        # 掩码 (1.0 - dones)
    'grasp_penalty': float # 抓取惩罚(可选)
}
```

### 分类器数据格式
```python
# 成功/失败数据文件包含转换列表
transitions = [
    {
        'observations': {
            'image': np.array,  # 图像
        },
        'labels': int  # 0=失败, 1=成功
    },
    ...
]
```

## 核心特性

### 1. 自动设备检测
```python
from pytorch_serl.utils.device import DEVICE
print(f"自动检测设备: {DEVICE}")  # cuda, mps, 或 cpu
```

### 2. SAC智能体
```python
from pytorch_serl.agents.sac import SACAgent, SACHybridAgent

# 标准SAC (固定抓取)
agent = SACAgent(
    image_keys=['image'],
    action_dim=6,
    lr=3e-4
)

# 混合SAC (学习抓取)
agent = SACHybridAgent(
    image_keys=['image'], 
    continuous_action_dim=6,
    grasp_action_dim=3,
    lr=3e-4
)
```

### 3. 奖励分类器
```python
from pytorch_serl.networks.classifier import RewardClassifier

classifier = RewardClassifier(image_keys=['image'])
success_prob, is_success = classifier.predict_success(obs)
```

## 训练配置

### 默认超参数
- **学习率**: 3e-4
- **折扣因子**: 0.95  
- **批次大小**: 256
- **目标熵**: -action_dim/2
- **软更新率**: 0.005
- **Critic集成大小**: 2

### 训练参数
- **训练开始步数**: 1000 (混合模式)
- **随机探索步数**: 1000 (混合模式) 
- **CTA比率**: 2 (Critic-to-Actor更新比率)
- **缓冲区容量**: 200,000

## 设备支持

代码自动检测并使用最佳可用设备：

1. **CUDA** (NVIDIA GPU) - 最高优先级
2. **MPS** (Apple Silicon) - 中等优先级  
3. **CPU** - 后备选项

## 注意事项

- 这是一个简化版本，专注于核心功能
- 移除了复杂的配置选项，使用固定的最佳实践参数
- 模拟环境用于测试，实际使用需要替换为真实环境接口
- 保持了与原始JAX版本相同的算法核心逻辑
