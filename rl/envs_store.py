from collections import deque
from typing import Dict, Optional
import os
import sys
import gymnasium as gym
import numpy as np
import numpy as np
import copy
from gymnasium import Env, spaces
import numpy as np
import time
from scipy.spatial.transform import Rotation
from gymnasium.spaces import flatten_space, flatten
from PIL import Image

sys.path.append(os.getcwd())
from serl_robot_infra.franka_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)
from serl_robot_infra.franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
)
from utils.rotations import euler_2_quat


def print_color(x):
    return print("\033[35m {}\033[00m".format(x))


class UR_Platform_Env(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        config=None,
    ):
        sys.path.append("/home/facelesswei/code/debug_UR_Robot_Arm_Show/tools")
        from zmq_tools import ZMQClient

        self.client = ZMQClient()
        self.action_scale = config.ACTION_SCALE
        self._RESET_POSE = config.RESET_POSE
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH

        self.gripper_open_pose = config.GRIPPER_OPEN_POSE
        self.gripper_close_pose = config.GRIPPER_CLOSE_POSE
        self.gripper_speed = config.GRIPPER_SPEED
        self.gripper_force = config.GRIPPER_FORCE
        self._update_currpos()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        # xyz + quat
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        # "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        # "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                        for key in config.REALSENSE_CAMERAS
                    }
                ),
            }
        )

        if fake_env:
            return

        self.cap = None
        self.reward = 0
        self.curr_path_length = 0

        print("Initialized UR")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        # print(f"self.xyz_bounding_box.low: {self.xyz_bounding_box.low}")
        # print(f"self.xyz_bounding_box.high: {self.xyz_bounding_box.high}")
        # print(f"pose[:3]: {pose[:3]}")

        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        # print(f"CLIP: pose[:3]: {pose[:3]}")
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.perf_counter()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        # print(f"[DEBUG] self.currpos: {self.currpos}")

        self.nextpos = self.currpos.copy()
        # print(f"delta action: {xyz_delta} :  {xyz_delta * self.action_scale[0]}")
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        #     * Rotation.from_quat(self.currpos[3:])
        # ).as_quat()
        self.nextpos[3:] = (
            euler_2_quat(self._RESET_POSE[3:])
            if self._RESET_POSE.shape[0] == 6
            else self._RESET_POSE[3:]
        )

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt_s = time.perf_counter() - start_time
        min_step_time = 1 / 20  # 20hz
        if dt_s < min_step_time:
            print(
                f"[UR_Platform_Env] sleep min_step_time: {(min_step_time - dt_s):.4f}s"
            )
            time.sleep(min_step_time - dt_s)

        self._update_currpos()
        ob = self._get_obs()
        reward = self.reward
        done = self.curr_path_length >= self.max_episode_length or reward
        if reward == 1:
            print_color(f"[UR_Platform_Env]: reward 1")
        if self.curr_path_length >= self.max_episode_length:
            # if executed time exceeds max length, give a -1 penalty.
            reward = 0
            print_color(f"[UR_Platform_Env]: max_episode_length reward {reward}")
        return ob, int(reward), done, False, {"succeed": reward}

    def step_long(self, target: np.ndarray) -> tuple:
        """standard gym step function."""

        while np.linalg.norm(target - self.currpos[:3]) > 0.03:
            diff = target - self.currpos[:3]

            normal_diff = diff / np.linalg.norm(diff)
            normal_diff *= 0.01
            print(f"diff: :{np.linalg.norm(target - self.currpos[:3])}")
            action = normal_diff + self.currpos[:3]
            action = np.concatenate([action, self.currpos[3:]])
            self._send_pos_command(action)
            self._update_currpos()

        observation, reward, terminated, truncated, info = self.step(np.zeros((7,)))

        return observation, reward, True, True, info

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}

        for key, image in self.cap.items():
            if key not in self.config.REALSENSE_CAMERAS.keys():
                continue

            # (480, 640, 3)
            rgb = image
            cropped_rgb = (
                self.config.IMAGE_CROP[key](rgb)
                if key in self.config.IMAGE_CROP
                else rgb
            )
            resized = np.array(
                Image.fromarray(cropped_rgb).resize(
                    self.observation_space["images"][key].shape[:2]
                )
            )
            images[key] = resized

        return images

    def go_to_reset(self):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        print("====JOINT RESET====")
        # requests.post(self.url + "jointreset")
        arr = np.array(self._RESET_POSE).astype(np.float32)
        if self.randomreset:
            noise = np.random.normal(0, 0.01, 3)
            arr[:3] += noise
        if self.gripper_open_pose:
            arr = np.concatenate([arr, [self.gripper_open_pose]])

        data = {"type": "jointreset", "arr": arr.tolist()}
        self.client.post(data)
        time.sleep(3)
        return

    def reset(self, joint_reset=False, **kwargs):
        print("[UR_Platform_Env] Resetting robot")

        self.go_to_reset()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        return obs, {"succeed": False}

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        # print(f"[DEBUG] _send_pos_command {pos}")

        arr = np.array(pos).astype(np.float32)
        data = {"type": "pose", "arr": arr.tolist()}
        self.client.post(data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""

        if mode == "binary":

            # print(f"[DEBUG] _send_g {pos}({self.currgripper})")

            if pos <= -0.5:  # close gripper
                self.client.post(
                    {
                        "type": "close_gripper",
                        "arr": self.gripper_close_pose,
                        "gripper_speed": self.gripper_speed,
                        "gripper_force": self.gripper_force,
                    }
                )

            elif pos >= 0.5:  # open gripper
                self.client.post(
                    {"type": "open_gripper", "arr": self.gripper_open_pose}
                )
            else:
                return

        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = self.client.post({"type": "getstate"})
        self.currpos = np.array(ps["pose"])
        self.currgripper = np.array(ps["gripper"])
        self.curr_force = np.array(ps["force"])
        # print(f"[DEBUG] curr_force: {self.curr_force}")

        self.cap = ps["obs"]
        if "reward" in ps["obs"]:
            self.reward = ps["obs"]["reward"]
        else:
            # print("[W] No reward in observation, set to 0")
            self.reward = 0

    def _get_obs(self) -> dict:
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_force": self.curr_force,
            "gripper_pose": self.currgripper,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()


class RecordEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
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
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
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
            "tcp_force": np.zeros((6,)),
            "gripper_pose": np.zeros((1,)),
        }
        self.fake_obs = dict(images=images, state=state_observation)
        self.curr_path_length = 0

    def step(self, action: np.ndarray) -> tuple:
        self.curr_path_length += 1
        done = self.curr_path_length >= 20
        # time.sleep(0.2)
        reward = 0

        info = {}
        info["intervene_action"] = np.random.uniform(-1, 1, size=(7,))
        info["succeed"] = reward
        return (
            copy.deepcopy(self.fake_obs),
            int(reward),
            done,
            False,
            info,
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
            theta_b_r = Rotation.from_matrix(T_b_r[:3, :3]).as_quat()
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
            (tcp_pose[:3], Rotation.from_quat(tcp_pose[3:]).as_euler("xyz"))
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
        proprio_list = []
        for key in self.proprio_keys:
            val = np.array(obs["state"][key], dtype=np.float32)
            # 如果是标量，变成 shape (1,)
            if val.ndim == 0:
                val = np.expand_dims(val, 0)
            assert val.ndim == 1

            proprio_list.append(val)

        proprio = np.concatenate(proprio_list, axis=-1)

        obs = {
            "state": proprio,
            **(obs["images"]),
        }
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


def stack_obs(obs_list):
    dict_list = {k: [dic[k] for dic in obs_list] for k in obs_list[0]}
    return {k: np.stack(v) for k, v in dict_list.items()}


def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()


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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, -1]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:  # FIXME need?
            action = info["intervene_action"]

        info["grasp_penalty"] = 0.0
        if action[-1] < -0.5:  # close gripper
            if self.last_gripper_pos == 1:
                print_color(
                    f"[GripperPenaltyWrapper] penalty {self.penalty} ({self.last_gripper_pos}:{action[-1]})"
                )
                info["grasp_penalty"] = self.penalty

        elif action[-1] > 0.5:  # open gripper
            if self.last_gripper_pos == 0:
                print_color(
                    f"[GripperPenaltyWrapper] penalty {self.penalty} ({self.last_gripper_pos}:{action[-1]})"
                )
                info["grasp_penalty"] = self.penalty
        else:
            pass

        self.last_gripper_pos = observation["state"][0, -1]
        return observation, reward, terminated, truncated, info


class Action7to4Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.action_space.shape == (7,)

        self.action_space = gym.spaces.Box(
            np.ones((4,), dtype=np.float32) * -1,
            np.ones((4,), dtype=np.float32),
        )

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        if len(action) == 4:
            action = [*action[:3], *[0.0, 0.0, 0.0], *action[-1:]]

        observation, reward, terminated, truncated, info = self.env.step(action)
        if len(action) == 7:
            action = [*action[:3], *action[-1:]]
        if "intervene_action" in info:
            if len(info["intervene_action"]) == 7:
                info["intervene_action"] = [
                    *info["intervene_action"][:3],
                    *info["intervene_action"][-1:],
                ]

        return observation, reward, terminated, truncated, info


class HumanRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        from pynput import keyboard

        self.success_key = False
        self.failure_key = False
        self.bad_key = False

        self.pressed_keys = None

        def on_press(key):
            try:
                if str(key) == "Key.space":
                    self._set_success()
                elif str(key) == "Key.ctrl_r":
                    self._set_failure()
                elif key.char == ".":
                    self._set_bad()
                else:
                    self.pressed_keys = key.char

            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def _set_success(self):
        print_color("Success key Pressed")
        self.success_key = True

    def _set_failure(self):
        print_color("Failure key Pressed")
        self.failure_key = True

    def _set_bad(self):
        print_color("Bad key Pressed")
        self.bad_key = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.success_key:
            self.success_key = False
            reward = 1.0
            done = True
            info["succeed"] = reward
        elif self.failure_key:
            self.failure_key = False
            reward = -0.2
            done = True
        elif self.bad_key:
            self.bad_key = False
            reward = -0.1
            done = False
        else:
            reward = 0.0 if int(reward) == 0 else reward

        """
        info["grasp_penalty"] = 0.0
        if self.pressed_keys is not None:
            print_color(f"pressed_keys: {self.pressed_keys}")
            self.pressed_keys = None

            if self.pressed_keys == "z":
                done = True
                info["grasp_penalty"] = -1.0
            elif self.pressed_keys == "x":
                done = True
                info["grasp_penalty"] = 1.0
        """

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.bad_key = False
        self.success_key = False
        self.failure_key = False
        return self.env.reset(**kwargs)

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        return self.env.close()


def get_fake_environment():
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
