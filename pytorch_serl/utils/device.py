"""设备配置工具，支持CUDA和MPS"""

import torch


def get_device():
    """自动检测并返回最佳设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


# 全局设备
DEVICE = get_device()
print(f"使用设备: {DEVICE}")
