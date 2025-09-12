import logging
import os
import sys
import torch


def print_dict_structure(d, indent=0):
    """递归打印字典结构"""
    for key, value in d.items():
        prefix = "      " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict_structure(value, indent + 1)
        else:
            if value is None:
                print(f"{prefix}{key}: None")
            elif hasattr(value, "shape"):
                print(f"{prefix}{key}: {value.shape}")
            else:
                print(f"{prefix}{key}: {value}")


def get_device():
    """自动检测并返回最佳设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(out_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(process)d %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
