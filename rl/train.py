import sys
import os

sys.path.append(os.getcwd())
from torchvision import transforms

# from examples.experiments.usb_pickup_insertion.ur_wrapper import (
#     Fake_UR_Platform_Env,
#     UR_Platform_Env,
# )
# from examples.experiments.usb_pickup_insertion.wrapper import HumanRewardEnv
# from serl_launcher.serl_launcher.wrappers.chunking import ChunkingWrapper
# from serl_launcher.serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
# from serl_robot_infra.franka_env.envs.relative_env import RelativeFrame
# from serl_robot_infra.franka_env.envs.wrappers import (
#     Quat2EulerWrapper,
#     SpacemouseIntervention,
# )
from utils.tools import print_dict_structure



def make_env(debug=True, fake_env=False):
    proprio_keys = ["tcp_pose", "gripper_pose"]

    if debug:
        env = Fake_UR_Platform_Env()
    else:
        env = UR_Platform_Env(fake_env=fake_env, config=UREnvConfig())
    if not debug:
        env = HumanRewardEnv(env)
    if not debug:
        env = SpacemouseIntervention(env)
    env = RelativeFrame(env, include_relative_pose=False)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    # env = GripperPenaltyWrapper(env, penalty=-0.02)
    return env


if __name__ == "__main__":
    env = make_env(debug=True)

    print("==== observation_space ====")
    print_dict_structure(env.observation_space)

    print("==== action_space ====")
    print(env.action_space)
