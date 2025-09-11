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
