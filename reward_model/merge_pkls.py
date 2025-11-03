import os
import pickle
import glob

from tqdm import tqdm


def main():

    output_dir = "datasets/plug_with_power_cord_10_31"
    os.makedirs(output_dir, exist_ok=True)

    # 定义要加载的文件列表
    data_files = [
        "/home/facelesswei/code/hil-serl-zbh/examples/experiments/usb_pickup_insertion/plug_with_power_cord/demo_buffer/*.pkl",
        "/home/facelesswei/code/hil-serl-zbh/examples/experiments/usb_pickup_insertion/plug_with_power_cord2/demo_buffer/*.pkl",
        "/home/facelesswei/code/hil-serl-zbh/examples/experiments/usb_pickup_insertion/plug_with_power_cord3/demo_buffer/*.pkl",
        "/home/facelesswei/code/hil-serl-zbh/examples/experiments/usb_pickup_insertion/plug_with_power_cord4/demo_buffer/*.pkl",
        "/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-10-27/usb_pickup_insertion_31_18-18-00.pkl",
    ]
    pkl_files = []
    for file_pattern in data_files:
        # 处理通配符模式
        if "*" in file_pattern:
            # 使用 glob 查找匹配的文件

            matched_files = glob.glob(file_pattern)

            if not matched_files:
                print(f"没有找到匹配的文件: {file_pattern}")
                continue
            for file_path in matched_files:
                pkl_files.append(file_path)
        else:
            pkl_files.append(file_pattern)

    if not pkl_files:
        print("目录下没有 pkl 文件")
        return

    all_steps = []

    for file_path in tqdm(pkl_files):
        with open(file_path, "rb") as f:
            steps = pickle.load(f)
            all_steps.extend(steps)  # 合并到总列表

    print(f"总共合并 {len(pkl_files)} 个文件，步骤数量: {len(all_steps)}")

    save_path = os.path.join(output_dir, f"merged_data{len(all_steps)}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(all_steps, f)

    print(f"已保存合并后的文件: {save_path}")


if __name__ == "__main__":
    main()
