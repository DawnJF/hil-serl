import os
import numpy as np
import sys
from rl.envs_store import *


class OpenSwitchEnvConfig:
    REALSENSE_CAMERAS = {
        "wrist": {"dim": (1280, 720)},
        "rgb": {"dim": (1280, 720)},
        "scene": {"dim": (1280, 720)},
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[20:330, 100:560],
        "rgb": lambda img: img[280:510, 150:490],
        "scene": lambda img: img[160:460, 100:560],
    }
    reset_euler = np.array([np.pi, 0, np.pi * 3 / 4])

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

    # reset_xyz = np.array([-0.6, -0.28, 0.1])
    reset_xyz = np.array([-0.55, -0.5, 0.14])
    reset_quat = np.array(
        [
            0.9797053086774433,
            0.20010938213011484,
            0.010719838360479872,
            0.004339170227621892,
        ]
    )
    RESET_POSE = np.array([*reset_xyz, *reset_quat])
    ACTION_SCALE = np.array([0.01, 0.02, 1])  # xyz, euler, gripper
    GRIPPER_OPEN_POSE = 170
    GRIPPER_CLOSE_POSE = 212
    MAX_EPISODE_LENGTH = 100


class OpenSwitchTrainConfig:
    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 256

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    # image_keys = ["wrist", "rgb", "scene"]
    image_keys = ["rgb", "scene"]
    classifier_keys = ["side_classifier"]
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio_keys = ["tcp_pose", "gripper_pose"]
    checkpoint_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000

    def get_environment(self, fake_env=False, debug=False):
        config = OpenSwitchEnvConfig()
        env = UR_Platform_Env(fake_env=fake_env, config=config)
        env = HumanRewardEnv(env)
        env = SpacemouseIntervention(env)
        env = RelativeFrame(env, include_relative_pose=False)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = GripperPenaltyWrapper(env, penalty=-0.04)
        return env
