import os
import time
import jax
import sys

sys.path.append(os.getcwd())
from examples.experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env
from rl.envs_temp import (
    ChunkingWrapper,
    GripperPenaltyWrapper,
    Quat2EulerWrapper,
    SERLObsWrapper,
    UREnvConfig,
    RelativeFrame,
)
from utils.tools import print_dict_structure
from bc.train_bc2rl import ActorWrapper

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map


def test_Env():

    ckpt_path = "outputs/bc2rl_20250928_173221/checkpoint-32.pth"
    model = ActorWrapper(ckpt_path)

    proprio_keys = ["tcp_pose", "gripper_pose"]
    config = UREnvConfig()
    config.MAX_EPISODE_LENGTH = 1000

    env = UR_Platform_Env(config=config)
    # env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=-0.02)

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
    test_Env()
