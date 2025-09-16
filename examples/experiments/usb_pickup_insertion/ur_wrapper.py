from typing import OrderedDict
from franka_env.utils.rotations import euler_2_quat, quat_2_euler
from franka_env.envs.franka_env import DefaultEnvConfig
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from typing import Dict
from scipy.spatial.transform import Rotation, Slerp
import sys
from PIL import Image

sys.path.append("/home/facelesswei/code/debug_UR_Robot_Arm_Show/tools")
from zmq_tools import ZMQClient


def pose_linspace(start_pose, goal_pose, steps):
    """
    插值位姿 (位置 + 四元数旋转)

    参数:
    - start_pose: np.ndarray, shape=(7,) [x, y, z, qx, qy, qz, qw]
    - goal_pose: np.ndarray, shape=(7,) 同上
    - steps: int, 插值的步数

    返回:
    - poses: np.ndarray, shape=(steps, 7)
    """

    start_pose = np.asarray(start_pose, dtype=float)
    goal_pose = np.asarray(goal_pose, dtype=float)

    # 1. 插值位置 (线性)
    xyz = np.linspace(start_pose[:3], goal_pose[:3], steps)

    # 2. 插值姿态 (四元数 -> SLERP)
    q1 = Rotation.from_quat(start_pose[3:])  # [x, y, z, w]
    q2 = Rotation.from_quat(goal_pose[3:])
    key_times = [0, 1]
    key_rots = Rotation.from_quat([q1.as_quat(), q2.as_quat()])
    slerp = Slerp(key_times, key_rots)

    times = np.linspace(0, 1, steps)
    quats = slerp(times).as_quat()

    # 3. 拼接结果
    poses = np.hstack([xyz, quats])
    return poses


class UR_Platform_Env(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        config: DefaultEnvConfig = None,
    ):
        self.client = ZMQClient()
        self.action_scale = config.ACTION_SCALE
        self._RESET_POSE = config.RESET_POSE
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH

        self.gripper_sleep = config.GRIPPER_SLEEP

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = (
            config.JOINT_RESET_PERIOD
        )  # reset the robot joint every 200 cycles

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
                        for key in config.REALSENSE_CAMERAS
                    }
                ),
            }
        )
        self.cycle_count = 0

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

        self.nextpos = self.currpos.copy()
        # print(f"delta action: {xyz_delta} :  {xyz_delta * self.action_scale[0]}")
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        #     * Rotation.from_quat(self.currpos[3:])
        # ).as_quat()
        self.nextpos[3:] = euler_2_quat(self._RESET_POSE[3:])

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt_s = time.perf_counter() - start_time
        min_step_time = 1 / 30  # 30hz
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
            print(f"\033[35m [UR_Platform_Env]: reward 1\033[0m")
        if self.curr_path_length >= self.max_episode_length:
            # if executed time exceeds max length, give a -1 penalty.
            reward = -1
            print(
                f"\033[34m[UR_Platform_Env]: max_episode_length {self.max_episode_length}\033[0m"
            )
        return ob, int(reward), done, False, {"succeed": reward}

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

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_currpos()

        path = pose_linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=True):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            # requests.post(self.url + "jointreset")
            arr = np.array(self._RESET_POSE).astype(np.float32)
            data = {"type": "jointreset", "arr": arr.tolist()}
            self.client.post(data)
            time.sleep(7)
            return

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

    def reset(self, joint_reset=False, **kwargs):
        print("[UR_Platform_Env] Resetting robot")
        self.last_gripper_act = time.time()

        self.cycle_count += 1
        if (
            self.joint_reset_cycle != 0
            and self.cycle_count % self.joint_reset_cycle == 0
        ):
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        # self.go_to_reset(joint_reset=joint_reset)
        self.go_to_reset()
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        return obs, {"succeed": False}

    def _recover(self):
        """Internal function to recover the robot from error state."""
        self.client.post({"type": "clearerr"})

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        # print(f"[DEBUG] _send_pos_command {pos}")
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"type": "pose", "arr": arr.tolist()}
        self.client.post(data)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        # print(f"[DEBUG] _send_gripper_command {pos}")
        print(f"[DEBUG] _send_gripper_command pos:{pos} currgripper: {self.currgripper}")
        if mode == "binary":
            if (
                (pos <= -0.5)
                # and (self.currgripper > 0.85)
                # and (self.currgripper <= 0.25)
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):  # close gripper
                self.client.post({"type": "close_gripper"})
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (
                (pos >= 0.5)
                # and (self.currgripper < 0.85)
                # and (self.currgripper > 0.25)
                and (time.time() - self.last_gripper_act > self.gripper_sleep)
            ):  # open gripper
                self.client.post({"type": "open_gripper"})
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
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
            "gripper_pose": self.currgripper,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()


if __name__ == "__main__":

    # test
    sys.path.append("/home/robot/code/hil-serl/examples")
    from experiments.usb_pickup_insertion.config import UREnvConfig

    env = UR_Platform_Env(config=UREnvConfig())

    env.client.post({"type": "close_gripper"})

    time.sleep(2)

    obs, info = env.reset()
    print(obs.keys())
    print(f"obs['images']['wrist'].shape: {obs['images']['wrist'].shape}")
    print(f"obs['state']['tcp_pose'].shape: {obs['state']['tcp_pose'].shape}")

    # Test action space
    # Test multiple steps
    for i in range(5):
        action = env.action_space.sample()
        print(f"test action: {action}")
        obs, reward, done, truncated, info = env.step(action)

        print("✓ Step function works")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")
        if done:
            obs, info = env.reset()
