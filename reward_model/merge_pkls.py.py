import os
import pickle
from glob import glob


def main():

    folder_path = "/media/xiamu/d72c2cdf-d882-4de8-8a3c-298f7bf4be67/Downloads/new/data"
    pkl_files = sorted(glob(os.path.join(folder_path, "*.pkl")))

    if not pkl_files:
        print("目录下没有 pkl 文件")
        return

    all_steps = []

    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            steps = pickle.load(f)
            all_steps.extend(steps)  # 合并到总列表

    print(f"总共合并 {len(pkl_files)} 个文件，步骤数量: {len(all_steps)}")

    save_path = os.path.join(folder_path, "merged_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_steps, f)

    print(f"已保存合并后的文件: {save_path}")


if __name__ == "__main__":
    main()
