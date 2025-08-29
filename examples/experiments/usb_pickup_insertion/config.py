import os
import time
import jax
import numpy as np
import jax.numpy as jnp
import sys

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig, FakeFrankaEnv
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

sys.path.append(os.getcwd())
sys.path.append("/home/robot/code/hil-serl")
sys.path.append("/home/robot/code/hil-serl/examples")
from tools import print_dict_structure

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import USBEnv, GripperPenaltyWrapper
from experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "wrist_2": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy": {
            "serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_classifier": {
            "serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[50:-200, 200:-200],
        "wrist_2": lambda img: img[:-200, 200:-200],
        "side_policy": lambda img: img[250:500, 350:650],
        "side_classifier": lambda img: img[270:398, 500:628],
    }
    TARGET_POSE = np.array(
        [0.553, 0.1769683108549487, 0.25097833796596336, np.pi, 0, -np.pi / 2]
    )
    RESET_POSE = TARGET_POSE + np.array([0, 0.03, 0.05, 0, 0, 0])
    ACTION_SCALE = np.array([0.015, 0.1, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.06, 0.05, 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.01, 0.03, 0.1, 0.1, 0.3])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 120


class UREnvConfig(DefaultEnvConfig):
    REALSENSE_CAMERAS = {
        "wrist": {
            "dim": (1280, 720),
        },
        "rgb": {
            "dim": (1280, 720),
        },
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[50:-200, 200:-200],
        "rgb": lambda img: img[:-200, 200:-200],
    }
    # TARGET_POSE = np.array(
    #     [0.553, 0.1769683108549487, 0.25097833796596336, np.pi, 0, -np.pi / 2]
    # )
    reset_xyz = np.array([-0.35, -0.5, 0.15])
    reset_euler = np.array([np.pi, 0, np.pi * 3 / 4])
    RESET_POSE = np.array([*reset_xyz, *reset_euler])
    ACTION_SCALE = np.array([0.01, 0.02, 1])  # xyz, euler, gripper
    # RANDOM_RESET = True
    RANDOM_RESET = False

    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = np.concatenate(
        [np.array([-0.3, -0.2, 0.35]), reset_euler + np.array([0.1, 0.1, 0.3])]
    )
    ABS_POSE_LIMIT_LOW = np.concatenate(
        [np.array([-0.6, -0.6, 0.055]), reset_euler - np.array([0.1, 0.1, 0.3])]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 120


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "rgb"]
    classifier_keys = ["side_classifier"]
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # env = USBEnv(fake_env=fake_env, save_video=save_video, config=UREnvConfig())
        env = UR_Platform_Env(fake_env=fake_env, config=UREnvConfig())
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # if classifier:
        #     classifier = load_classifier_func(
        #         key=jax.random.PRNGKey(0),
        #         sample=env.observation_space.sample(),
        #         image_keys=self.classifier_keys,
        #         checkpoint_path=os.path.abspath("classifier_ckpt/"),
        #     )

        #     def reward_func(obs):
        #         sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        #         return int(sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 0] > 0.4)

        #     env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env


def test_mouse():
    env = UR_Platform_Env(fake_env=False, config=UREnvConfig())
    env = SpacemouseIntervention(env)

    print(f"action_space: {env.action_space}")

    time.sleep(1)
    env.reset()

    while True:
        action = env.action_space.sample()
        action = np.zeros((7,))
        print(f"test action: {action}")
        obs, reward, done, truncated, info = env.step(action)


def test_fake_Env():

    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    env = FakeFrankaEnv(config=EnvConfig())
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=-0.02)

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

    print("==== step ====")
    obs, reward, done, truncated, info = env.step(env.action_space.sample())

    print(obs.keys())
    print_dict_structure(obs)


if __name__ == "__main__":
    # test

    # test_fake_Env()
    test_mouse()
