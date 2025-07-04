"""重放缓冲区的PyTorch实现"""

import numpy as np
import torch
from collections import deque
import random
from typing import Dict, List, Tuple, Optional


class ReplayBuffer:
    """简单的重放缓冲区实现"""

    def __init__(self, capacity: int, image_keys: Optional[List[str]] = None):
        self.capacity = capacity
        self.image_keys = image_keys or []

        # 使用deque进行高效的插入和删除
        self.buffer = deque(maxlen=capacity)
        self._size = 0

    def insert(self, transition: Dict):
        """插入一个转换"""
        self.buffer.append(transition)
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """采样批次数据"""
        if self._size < batch_size:
            raise ValueError(f"缓冲区大小 {self._size} 小于批次大小 {batch_size}")

        # 随机采样
        batch_transitions = random.sample(self.buffer, batch_size)

        # 组织批次数据
        batch = {}

        # 处理观测
        batch["observations"] = self._collate_observations(
            [t["observations"] for t in batch_transitions], device
        )
        batch["next_observations"] = self._collate_observations(
            [t["next_observations"] for t in batch_transitions], device
        )

        # 处理其他字段
        batch["actions"] = torch.tensor(
            [t["actions"] for t in batch_transitions],
            dtype=torch.float32,
            device=device,
        )
        batch["rewards"] = torch.tensor(
            [t["rewards"] for t in batch_transitions],
            dtype=torch.float32,
            device=device,
        )
        batch["masks"] = torch.tensor(
            [t.get("masks", 1.0 - t.get("dones", False)) for t in batch_transitions],
            dtype=torch.float32,
            device=device,
        )

        # 处理可选字段
        if any("grasp_penalty" in t for t in batch_transitions):
            batch["grasp_penalty"] = torch.tensor(
                [t.get("grasp_penalty", 0.0) for t in batch_transitions],
                dtype=torch.float32,
                device=device,
            )

        if any("labels" in t for t in batch_transitions):
            batch["labels"] = torch.tensor(
                [t["labels"] for t in batch_transitions],
                dtype=torch.float32,
                device=device,
            )

        return batch

    def _collate_observations(
        self, obs_list: List[Dict], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """整理观测数据"""
        batch_obs = {}

        # 获取所有键
        all_keys = set()
        for obs in obs_list:
            all_keys.update(obs.keys())

        for key in all_keys:
            values = []
            for obs in obs_list:
                if key in obs:
                    values.append(obs[key])
                else:
                    # 如果某个观测中没有这个键，使用零填充
                    if values:
                        values.append(np.zeros_like(values[0]))
                    else:
                        continue

            if values:
                if key in self.image_keys:
                    # 图像数据，确保是CHW格式
                    values = np.array(values)
                    if len(values.shape) == 4 and values.shape[-1] == 3:  # BHWC -> BCHW
                        values = values.transpose(0, 3, 1, 2)
                    batch_obs[key] = torch.tensor(
                        values, dtype=torch.float32, device=device
                    )
                else:
                    # 其他数据
                    batch_obs[key] = torch.tensor(
                        np.array(values), dtype=torch.float32, device=device
                    )

        return batch_obs

    def __len__(self):
        return self._size

    def get_iterator(self, batch_size: int, device: torch.device):
        """获取数据迭代器"""

        def iterator():
            while True:
                if len(self) >= batch_size:
                    yield self.sample(batch_size, device)
                else:
                    # 如果数据不够，等待或返回空
                    break

        return iterator()
