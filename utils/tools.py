import logging
import os
import sys
import torch
import pickle as pkl


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
        level=logging.INFO,
        format="%(asctime)s %(process)d %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def inspect_pickle_structure(file_path, indent=0, max_depth=5):
    """
    打印 pickle 文件中对象的 key 结构（支持嵌套 dict / list / tuple）

    参数:
        file_path: str 或 Path - pickle 文件路径
        indent: int - 当前缩进级别（用于递归）
        max_depth: int - 最大递归深度（防止过深结构打印过多）
    """

    def _print_structure(obj, level=0):
        prefix = "    " * level
        if level > max_depth:
            print(f"{prefix}... (max depth reached)")
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                print(f"{prefix}📦 {k} ({type(v).__name__})")
                _print_structure(v, level + 1)
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}[{len(obj)} elements] ({type(obj).__name__})")
            if len(obj) > 0:
                print(f"{prefix}└─ first element:")
                _print_structure(obj[0], level + 1)
        else:
            print(f"{prefix}{type(obj).__name__}")

    file_path = os.path.abspath(file_path)
    with open(file_path, "rb") as f:
        obj = pkl.load(f)

    print(f"✅ Loaded pickle: {file_path}")
    print("📂 Structure:")
    _print_structure(obj)


if __name__ == "__main__":
    # 示例用法
    # test_pickle_path = "/home/facelesswei/code/hil-serl-debug/outputs/torch_rlpd/dddd/20251017-0943/demo_buffer/transitions_2000.pkl"
    test_pickle_path = "/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-09/usb_pickup_insertion_30_11-50-21.pkl"
    inspect_pickle_structure(test_pickle_path)
