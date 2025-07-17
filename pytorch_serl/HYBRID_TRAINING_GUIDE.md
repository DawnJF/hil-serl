# 混合模式训练使用手册

## 概述

这个混合模式训练系统实现了结合演示数据和在线强化学习的Human-in-the-Loop SERL训练方法。系统采用50/50采样策略，确保演示数据和在线数据在训练过程中始终占据固定比例。

## 主要特性

- **双线程架构**: 分离的演员(Actor)和学习者(Learner)线程，实现高效的并行训练
- **自适应采样**: 根据可用数据动态调整在线/演示数据比例
- **人工干预支持**: 支持在线人工干预和纠正
- **混合SAC智能体**: 结合连续动作和离散抓取动作的混合动作空间
- **实时网络更新**: 演员网络参数实时更新，保持与学习者同步

## 快速开始

### 1. 基础测试

运行快速测试验证安装：

```bash
cd /path/to/pytorch_serl
python test_and_demo_hybrid.py --test quick
```

### 2. 创建测试数据

如果没有演示数据，可以创建测试数据：

```bash
python test_and_demo_hybrid.py --create_data
```

这将在 `test_data/` 目录下创建：
- `success_demo.pkl`: 成功演示数据
- `failure_demo.pkl`: 失败演示数据

### 3. 运行完整训练

```bash
python test_and_demo_hybrid.py --test full --max_steps 10000
```

## 详细使用

### 使用自己的演示数据

```bash
python train_hybrid.py \
    --demo_paths /path/to/demo1.pkl /path/to/demo2.pkl \
    --image_keys image \
    --setup_mode single-arm-fixed-gripper \
    --max_steps 100000 \
    --batch_size 256 \
    --lr 3e-4 \
    --save_dir ./my_training_results
```

### 演示数据格式

演示数据应该是包含以下格式转换的pickle文件：

```python
transition = {
    "observations": {
        "image": np.array([H, W, C])  # 图像数据，HWC格式
    },
    "actions": np.array([7])          # 动作向量
    "next_observations": {
        "image": np.array([H, W, C])  # 下一个图像
    },
    "rewards": float,                 # 奖励值
    "masks": float,                   # 掩码 (1.0 - done)
    "dones": bool                     # 是否结束
}
```

### 参数配置

#### 关键参数

- `--max_steps`: 最大训练步数（默认：100000）
- `--batch_size`: 批次大小（默认：256）
- `--training_starts`: 开始训练的最小样本数（默认：1000）
- `--random_steps`: 随机探索步数（默认：1000）
- `--cta_ratio`: Critic-to-Actor更新比率（默认：2）

#### 设置模式

- `single-arm-fixed-gripper`: 单臂固定夹爪（动作维度：6）
- `single-arm-learned-gripper`: 单臂学习夹爪（动作维度：7）
- `dual-arm-fixed-gripper`: 双臂固定夹爪（动作维度：12）
- `dual-arm-learned-gripper`: 双臂学习夹爪（动作维度：14）

## 高级用法

### 自定义环境

要使用自己的环境，替换 `MockEnvironment`：

```python
from your_env import YourEnvironment

# 在 train_hybrid.py 中修改
env = YourEnvironment(image_keys, action_dim)
```

环境需要实现：
- `reset()`: 返回 (observation, info)
- `step(action)`: 返回 (next_obs, reward, done, truncated, info)

### 人工干预

在环境的 `step` 方法中添加人工干预逻辑：

```python
def step(self, action):
    # 正常执行动作
    next_obs, reward, done, truncated, info = self._step(action)
    
    # 检查人工干预
    if human_intervention_detected():
        corrected_action = get_human_action()
        info["intervene_action"] = corrected_action
    
    return next_obs, reward, done, truncated, info
```

## 训练监控

### 实时指标

训练过程中会显示以下指标：
- `critic_loss`: Critic网络损失
- `grasp_critic_loss`: 抓取Critic损失
- `actor_loss`: Actor网络损失
- `alpha_loss`: 温度参数损失
- `alpha`: 当前温度参数值

### 保存的文件

训练完成后，在保存目录中会找到：
- `final_model.pth`: 最终训练模型
- `checkpoint_*.pth`: 定期保存的检查点（每10000步）

### 加载模型

```python
import torch
from agents.sac import SACHybridAgent

# 创建智能体
agent = SACHybridAgent(
    image_keys=["image"],
    continuous_action_dim=6,
    grasp_action_dim=3,
    device="cuda"
)

# 加载模型
checkpoint = torch.load("path/to/final_model.pth")
agent.load(checkpoint)
```

## 性能优化

### 建议配置

对于不同规模的训练：

**快速测试**:
```bash
--max_steps 1000 --batch_size 32 --training_starts 100
```

**中等规模**:
```bash
--max_steps 50000 --batch_size 128 --training_starts 1000
```

**完整训练**:
```bash
--max_steps 500000 --batch_size 256 --training_starts 5000
```

### 内存优化

- 减少 `batch_size` 如果遇到内存不足
- 调整缓冲区容量在 `SharedReplayBuffer` 初始化中
- 使用更小的图像分辨率

### GPU加速

确保CUDA可用并设置正确的设备：

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 故障排除

### 常见问题

1. **设备不匹配错误**
   - 确保所有张量在同一设备上
   - 检查图像数据格式（CHW vs HWC）

2. **采样失败**
   - 检查缓冲区是否有足够数据
   - 验证演示数据格式正确

3. **训练不收敛**
   - 调整学习率
   - 检查奖励函数设计
   - 增加演示数据质量

### 调试模式

启用详细日志：

```bash
python train_hybrid.py --log_interval 10  # 更频繁的日志输出
```

## 扩展开发

### 添加新的网络架构

1. 在 `networks/` 目录下添加新网络
2. 修改 `SACHybridAgent` 以使用新网络
3. 更新保存/加载逻辑

### 集成新的采样策略

修改 `learner_loop` 中的采样逻辑：

```python
# 自定义采样策略
if custom_sampling_condition():
    online_batch_size = custom_online_size()
    demo_batch_size = custom_demo_size()
```

## 技术支持

如果遇到问题：

1. 首先运行测试套件验证安装
2. 检查演示数据格式
3. 查看训练日志中的错误信息
4. 参考示例配置文件

更多详细信息，请参考代码注释和测试文件。
