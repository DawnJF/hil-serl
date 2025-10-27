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
sys.path.append("/home/facelesswei/code/hil-serl-zbh")
sys.path.append("/home/facelesswei/code/hil-serl-zbh/examples")
from utils.tools import print_dict_structure

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import (
    HumanRewardEnv,
    USBEnv,
    GripperPenaltyWrapper,
    ImageTransformWrapper,
)
from experiments.usb_pickup_insertion.ur_wrapper import (
    UR_Platform_Env,
    Fake_UR_Platform_Env,
)


class UREnvConfig(DefaultEnvConfig):
    REALSENSE_CAMERAS = {
        "wrist": {
            "dim": (1280, 720),
        },
        "rgb": {
            "dim": (1280, 720),
        },
        "scene": {
            "dim": (1280, 720)
        }
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[20:330, 100:560],
        "rgb": lambda img: img[280:510, 150:490],
        "scene": lambda img: img[160:460, 100:560],
    }
    # TARGET_POSE = np.array(
    #     [0.553, 0.1769683108549487, 0.25097833796596336, np.pi, 0, -np.pi / 2]
    # )
    # reset_xyz = np.array([-0.35, -0.5, 0.15])
    # For iphone
    reset_xyz = np.array([
        -0.5999861359596252,
        -0.20111770927906036,
        0.12716920274496078,
    ])
    reset_euler = np.array([np.pi, 0, np.pi * 3 / 4])
    reset_quat = np.array([
        0.9797053086774433,
        0.20010938213011484,
        0.010719838360479872,
        0.004339170227621892
    ])
    RESET_POSE = np.array([*reset_xyz, *reset_quat])
    ACTION_SCALE = np.array([0.003, 0.02, 1])  # xyz, euler, gripper
    GRIPPER_OPEN_POSE = 170
    GRIPPER_CLOSE_POSE = 205
    GRIPPER_SPEED = 10
    GRIPPER_FORCE = 10
    RANDOM_RESET = True
    # RANDOM_RESET = False

    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    # [-0.5, -0.2, 0.25]
    # [-0.6, -0.6, 0.055]
    ABS_POSE_LIMIT_HIGH = np.concatenate(
        [np.array([-0.35, -0.15, 0.15]), reset_euler + np.array([0.1, 0.1, 0.3])]
    )
    ABS_POSE_LIMIT_LOW = np.concatenate(
        [np.array([-0.65, -0.55, 0.073]), reset_euler - np.array([0.1, 0.1, 0.3])]
    )
    MAX_EPISODE_LENGTH = 300
     
    # image transform configs
    TFS = {
        "brightness":{
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"brightness": [0.8, 1.2]}
        },
        "contrast": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"contrast": [0.8, 1.2]}
        },
        "saturation": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"saturation": [0.5, 1.5]}
        },
        "hue": {
            "weight": 1.0,
            "type": "ColorJitter",
            "kwargs": {"hue": [-0.05, 0.05]}
        },
        "sharpness": {
            "weight": 1.0,
            "type": "SharpnessJitter",
            "kwargs": {"sharpness": [0.5, 1.5]}
        },
        "translation":{
            "weight": 1.0,
            "type": "RandomAffine",
            "kwargs": {"degrees": 0, "translate": (0.1, 0.1)}
        },
        # "perspective":{
        #     "weight": 1.0,
        #     "type": "RandomPerspective",
        #     "kwargs": {
        #         "distortion_scale": 0.2,  # 中等变形强度，不破坏特征
        #         "p": 0.5,                 # 50%概率应用，平衡多样性和稳定性
        #         "fill": (0, 0, 0)         # 空白区域填黑色（根据你的数据集背景色调整）
        #     }
        # }
    }
    MAX_NUM_TRANSFORMS = 5  # maximum number of transforms to apply
    ENABLE_TRANSFORMS = True  # whether to enable image transforms
    RANDOM_ORDER = True  # whether to apply transforms in random order
    CAMERA_SECTIONS = ["wrist", "rgb", "scene"]
    PROBABILITY = 0.5  # probability to apply image transforms


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "rgb", "scene"]
    classifier_keys = ["side_classifier"]
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]
    checkpoint_period = 1000
    cta_ratio = 2
    random_steps = 0
    # discount = 0.99
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, debug=False):
        # env = USBEnv(fake_env=fake_env, save_video=save_video, config=UREnvConfig())
        env = UR_Platform_Env(fake_env=fake_env, config=UREnvConfig())
        env = HumanRewardEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env, include_relative_pose=False)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        #     env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.04)
        env = ImageTransformWrapper(env, config=UREnvConfig())
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


def test_images():
    from PIL import Image
    import jax

    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree.map
    if not hasattr(jax, "tree_leaves"):
        jax.tree_leaves = jax.tree.leaves
    
    proprio_keys = ["tcp_pose", "gripper_pose"]

    env = UR_Platform_Env(fake_env=False, config=UREnvConfig())
    env = RelativeFrame(env, include_relative_pose=False)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=-0.02)
    env = ImageTransformWrapper(env, config=UREnvConfig())
    env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(obs.keys())

    ##------ debug
    rgb = obs["rgb"]
    rgb_shape = rgb.shape
    print("rgb_shape", rgb_shape)
    print("rgb_type", type(rgb))

    wrist = obs["wrist"]
    wrist_shape = wrist.shape
    print("wrist_shape", wrist_shape)
    print("wrist_type", type(wrist))

    scene = obs["scene"]
    scene_shape = scene.shape
    print("scene_shape", scene_shape)
    print("scene_type", type(scene))

    ##------ debug
    Image.fromarray(rgb.squeeze(0)).save("outputs/test_rgb/rgb.png")
    Image.fromarray(wrist.squeeze(0)).save("outputs/test_rgb/wrist.png")
    Image.fromarray(scene.squeeze(0)).save("outputs/test_rgb/scene.png")


def test_reward_model():
    env = UR_Platform_Env(fake_env=False, config=UREnvConfig())
    env = HumanRewardEnv(env)
    env = SpacemouseIntervention(env)

    print(f"action_space: {env.action_space}")

    time.sleep(1)
    env.reset()

    while True:
        action = env.action_space.sample()
        action = np.zeros((7,))
        obs, reward, done, truncated, info = env.step(action)
        print(f"left: {info['left']}, right: {info['right']}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        if done:
            env.reset()


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
    # test_mouse()
    test_reward_model()
    # test_images()
