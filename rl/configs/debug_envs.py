import time
import sys
import os
import pickle as pkl

sys.path.append(os.getcwd())
from rl import mappings
from utils.tools import print_dict_structure


def debug_config():
    env = mappings.CONFIG_MAPPING["open_switch"]().get_environment()

    print("==== observation_space ====")
    print_dict_structure(env.observation_space)

    print("==== action_space ====")
    print(env.action_space)

    print("==== reset ====")

    obs, info = env.reset()
    print(obs.keys())
    print_dict_structure(obs)
    # print(f"obs['images']['wrist'].shape: {obs['images']['wrist'].shape}")
    # print(f"obs['state']['tcp_pose'].shape: {obs['state']['tcp_pose'].shape}")

    # image_w = Image.fromarray(obs["images"]["wrist"].squeeze())
    # image_w.save("/home/facelesswei/code/hil-serl/image_w.png")
    # image_r = Image.fromarray(obs["images"]["rgb"].squeeze())
    # image_r.save("/home/facelesswei/code/hil-serl/image_r.png")

    print("==== step ====")
    for _ in range(11110):
        start_time = time.perf_counter()
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        print(action)
        dt_s = time.perf_counter() - start_time
        print(f"dt_s: {dt_s}")


def replay_dataset():
    path = "/home/facelesswei/code/hil-serl/datasets/trajectories/2025-10-29/traj_19-22-40_16.pkl"

    env = mappings.CONFIG_MAPPING["open_switch"]().get_environment()

    env.reset()

    with open(path, "rb") as f:
        transitions = pkl.load(f)
        print(transitions[0].keys())

        for i in range(len(transitions)):
            print("==== step ", i, " ====")
            print(f"reward: {transitions[i]['rewards']}")
            print(f"done: {transitions[i]['dones']}")
            print(f"action: {transitions[i]['actions']}")
            print(f"masks: {transitions[i]['masks']}")

            env.step(transitions[i]["actions"])
            if transitions[i]["dones"]:
                print(f"Episode {i} finished.")
                break


if __name__ == "__main__":
    # debug_config()
    replay_dataset()
