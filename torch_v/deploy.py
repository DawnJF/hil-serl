import os
import time
import sys

import jax

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "examples")))
from examples.experiments.mappings import CONFIG_MAPPING
from utils.tools import print_dict_structure
from torch_v.train_bc2rl import ActorWrapper


def test_Env():

    ckpt_path = "outputs/bc2rl/20251103_171947/checkpoint-56.pth"
    model = ActorWrapper(ckpt_path)

    # config = UREnvConfig()
    task_name = "plug_into_socket_with_power_cord"
    config = CONFIG_MAPPING[task_name]()
    env = config.get_environment(train=False)

    print("==== observation_space ====")
    print_dict_structure(env.observation_space)

    print("==== action_space ====")
    print(env.action_space)

    obs, info = env.reset()
    print(obs.keys())
    print_dict_structure(obs)

    print("==== step ====")
    for _ in range(11110):
        start_time = time.perf_counter()

        action = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            obs, info = env.reset()

        # print(obs.keys())
        # print_dict_structure(obs)


if __name__ == "__main__":

    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree.map
    if not hasattr(jax, "tree_leaves"):
        jax.tree_leaves = jax.tree.leaves

    test_Env()
