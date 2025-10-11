import logging
import os
import sys
import torch


def logging_args(args, name=""):
    # 格式化打印参数，提高可读性
    logging.info("=" * 50)
    logging.info(f"{name} Parameters:")
    logging.info("=" * 50)
    for key, value in vars(args).items():
        logging.info(f"  {key:20}: {value}")
    logging.info("=" * 50)


def print_dict_device(d, indent=0):
    """递归打印字典结构"""
    for key, value in d.items():
        prefix = "      " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict_device(value, indent + 1)
        else:
            if value is None:
                print(f"{prefix}{key}: None")
            elif hasattr(value, "device"):
                print(f"{prefix}{key}: {value.device}")
            else:
                print(f"{prefix}{key}: {value}")


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
