import threading
import time
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import multiprocessing as mp

from rl.sac_policy import get_eval_transform
from rl.sac_policy import dict_data_to_torch


# -----------------------------
# 共享队列 + Dataset 定义
# -----------------------------
class SharedQueueDataset(IterableDataset):
    def __init__(self, queue: mp.Queue):
        self.queue = queue
        self._image_transform = None

    def _get_transform(self):
        # 延迟初始化，在worker进程中创建
        if self._image_transform is None:
            self._image_transform = get_eval_transform()
        return self._image_transform

    def __iter__(self):
        while True:
            batch = self.queue.get()  # 阻塞式取数据
            batch = dict_data_to_torch(batch, self._get_transform())
            yield batch


def get_shared_queue_iterator(fun_get_data):

    mp.set_start_method("spawn", force=True)  # 推荐在多进程中使用 spawn

    # 创建共享队列（可以被多个 worker 共享）
    shared_queue = mp.Queue(maxsize=50)

    # 创建 Dataset + DataLoader（4 个 worker 并行消费）
    dataset = SharedQueueDataset(shared_queue)
    dataloader = DataLoader(dataset, num_workers=6, batch_size=None)

    # 启动 DataLoader 的迭代器
    data_iter = iter(dataloader)

    # 启动一个线程不断往队列里放数据
    def data_producer():
        while shared_queue.qsize() < 10:
            data = fun_get_data()
            shared_queue.put(data)  # 阻塞式放数据
            time.sleep(0.01)  # 模拟数据生成时间

    # use threading.Thread

    producer_thread = threading.Thread(target=data_producer, daemon=True)
    producer_thread.start()

    return shared_queue, data_iter


if __name__ == "__main__":

    shared_queue, data_iter = get_shared_queue_iterator()
    # 主训练循环

    add_size = 10 - shared_queue.qsize()
    if add_size > 0:
        print(f"[Main] Queue size={shared_queue.qsize()} → Filling...")
        for _ in range(add_size):
            # 模拟生成数据
            dummy_data = {}
            shared_queue.put(dummy_data)

    # ---- 消费者逻辑（DataLoader worker 自动在取） ----
    batch = next(data_iter)
