import pickle
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


def show_frame(steps, frame_idx, close_key="b"):
    """显示某一帧图像"""
    frame_idx = int(frame_idx)
    if frame_idx < 0 or frame_idx >= len(steps):
        print(f"帧索引 {frame_idx} 超出范围，最大索引 {len(steps) - 1}")
        return

    rgb = steps[frame_idx]["observations"]["rgb"]
    if rgb.ndim == 4:
        rgb = rgb[0]

    fig, ax = plt.subplots()
    ax.imshow(rgb.astype(np.uint8))
    ax.set_title(f"Step {frame_idx}")
    ax.axis("off")
    print(f"按 '{close_key}' 键关闭图像窗口")

    def on_key(event):
        if event.key == close_key:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def modify_rewards(steps, mode, save_path, index=None):
    """修改 steps 的 rewards 并立即保存"""
    if mode == "0":
        for s in steps:
            s["rewards"] = 0
    elif mode == "1":
        for s in steps:
            s["rewards"] = 1
    elif mode == "01":
        if index is None:
            print("请提供 index")
            return
        while True:
            index = int(index)
            if 0 <= index < len(steps):
                break
            else:
                print(f"分割索引 {index} 超出范围，最大索引 {len(steps) - 1}")
                index = input("请重新输入分割索引 index: ")

        for i, s in enumerate(steps):
            s["rewards"] = 0 if i < index else 1

    print("修改后的前几个 rewards:", [s["rewards"] for s in steps[:10]], "...")

    with open(save_path, "wb") as f:
        pickle.dump(steps, f)
    print(f"已保存标注文件: {save_path}")


def main():
    # 设置 pkl 文件目录
    folder_path = "/home/facelesswei/code/hil-serl/classifier_data/2025-09-10/"
    pkl_files = sorted(glob(os.path.join(folder_path, "*.pkl")))

    if not pkl_files:
        print("目录下没有 pkl 文件")
        return

    file_idx = 0
    while file_idx < len(pkl_files):
        file_path = pkl_files[file_idx]
        with open(file_path, "rb") as f:
            steps = pickle.load(f)
        save_path = file_path
        # save_path = file_path.replace(".pkl", "_labeled.pkl")
        print(f"\n[{file_idx}] 当前文件: {os.path.basename(file_path)}")
        print(f"该文件共有 {len(steps)} 个 step")

        while True:
            c = input(
                "输入帧索引 或 指令 (#0, #1, #01(前部分置0 后部分置1), q退出): "
            ).strip()
            if c.lower() == "q":
                return
            elif c.startswith("#"):
                parts = c[1:]
                if parts in ["0", "1"]:
                    modify_rewards(steps, parts, save_path)
                    break
                elif parts == "01":
                    idx = input("请输入分割索引 index: ")
                    modify_rewards(steps, "01", save_path, idx)
                    break
                else:
                    print("无效指令")
            else:
                try:
                    frame_idx = int(c)
                    show_frame(steps, frame_idx, close_key="b")
                except ValueError:
                    print("输入错误，请输入帧索引 或有效指令")

        file_idx += 1

    print("所有文件已处理完毕！")


if __name__ == "__main__":
    main()
