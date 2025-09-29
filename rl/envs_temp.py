from collections import deque
from typing import Dict, Optional
import os
import sys
import gymnasium as gym
import numpy as np
import numpy as np
import copy
import gymnasium as gym
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
import time
import copy
from scipy.spatial.transform import Rotation as R
from gymnasium.spaces import flatten_space, flatten

sys.path.append(os.getcwd())
from examples.experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env
from examples.experiments.usb_pickup_insertion.wrapper import HumanRewardEnv
from serl_launcher.serl_launcher.wrappers.chunking import space_stack
from serl_robot_infra.franka_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)
from serl_robot_infra.franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
)


class Fake_UR_Platform_Env(gym.Env):

    def __init__(self):
        REALSENSE_CAMERAS = {
            "wrist": {
                "dim": (1280, 720),
            },
            "rgb": {
                "dim": (1280, 720),
            },
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in REALSENSE_CAMERAS
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        images = {
            "wrist": np.zeros((128, 128, 3), dtype=np.uint8),
            "rgb": np.zeros((128, 128, 3), dtype=np.uint8),
        }
        state_observation = {
            "tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1]),
            "gripper_pose": np.zeros((1,)),
        }
        self.fake_obs = dict(images=images, state=state_observation)
        self.curr_path_length = 0

    def step(self, action: np.ndarray) -> tuple:
        self.curr_path_length += 1
        done = self.curr_path_length >= 20
        # time.sleep(0.2)
        reward = 0
        return (
            copy.deepcopy(self.fake_obs),
            int(reward),
            done,
            False,
            {"succeed": reward},
        )

    def reset(self, **kwargs):
        self.curr_path_length = 0
        time.sleep(0.2)
        return copy.deepcopy(self.fake_obs), {"succeed": False}


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["original_state_obs"] = copy.deepcopy(obs["state"])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        # obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix.
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def stack_obs(obs_list):
    dict_list = {k: [dic[k] for dic in obs_list] for k in obs_list[0]}
    return {k: np.stack(v) for k, v in dict_list.items()}


class ChunkingWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon
            )

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon

        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)
        return (stack_obs(self.current_obs), reward, done, trunc, info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

        self.action_space = gym.spaces.Box(
            np.ones((4,), dtype=np.float32) * -1,
            np.ones((4,), dtype=np.float32),
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        if len(action) == 4:
            action = [*action[:3], *[0.0, 0.0, 0.0], *action[-1:]]

        observation, reward, terminated, truncated, info = self.env.step(action)
        if len(action) == 7:
            action = [*action[:3], *action[-1:]]
        if "intervene_action" in info:
            action = info["intervene_action"]
            if len(action) == 7:
                info["intervene_action"] = [*action[:3], *action[-1:]]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        # self.last_gripper_pos = observation["state"][0, 0]
        self.last_gripper_pos = action[-1]
        return observation, reward, terminated, truncated, info


def get_fake_environment(fake_env=False, save_video=False, debug=False):
    proprio_keys = ["tcp_pose", "gripper_pose"]

    env = Fake_UR_Platform_Env()
    env = RelativeFrame(env, include_relative_pose=False)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    #     env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
    env = GripperPenaltyWrapper(env, penalty=-0.02)
    # env = ImageTransformWrapper(env, config=UREnvConfig())
    return env


class UREnvConfig:
    SERVER_URL: str = "http://127.0.0.1:5000/"
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0
    REALSENSE_CAMERAS = {
        "wrist": {
            "dim": (1280, 720),
        },
        "rgb": {
            "dim": (1280, 720),
        },
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[0:300, 0:640],
        "rgb": lambda img: img[300:420, 390:640],
    }
    reset_xyz = np.array([-0.35, -0.5, 0.15])
    reset_euler = np.array([np.pi, 0, np.pi * 3 / 4])
    RESET_POSE = np.array([*reset_xyz, *reset_euler])
    ACTION_SCALE = np.array([0.006, 0.02, 1])  # xyz, euler, gripper
    RANDOM_RESET = False

    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = np.concatenate(
        [np.array([-0.3, -0.2, 0.25]), reset_euler + np.array([0.1, 0.1, 0.3])]
    )
    ABS_POSE_LIMIT_LOW = np.concatenate(
        [np.array([-0.6, -0.6, 0.055]), reset_euler - np.array([0.1, 0.1, 0.3])]
    )
    MAX_EPISODE_LENGTH = 200

    MAX_NUM_TRANSFORMS = 7  # maximum number of transforms to apply
    ENABLE_TRANSFORMS = True  # whether to enable image transforms
    RANDOM_ORDER = True  # whether to apply transforms in random order
    CAMERA_SECTIONS = ["wrist", "rgb"]
    PROBABILITY = 0.5  # probability to apply image transforms


def get_environment(fake_env=False, debug=False):
    proprio_keys = ["tcp_pose", "gripper_pose"]

    env = UR_Platform_Env(fake_env=fake_env, config=UREnvConfig())

    env = HumanRewardEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env, include_relative_pose=False)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = GripperPenaltyWrapper(env, penalty=-0.02)
    return env


if __name__ == "__main__":
    env = get_environment()
    print(env)
