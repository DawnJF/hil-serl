from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from franka_env.envs.franka_env import FrankaEnv

# for image transformation
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
import collections
from typing import Any, Callable, Sequence
import torch
from torchvision.transforms.v2 import functional as F
from franka_env.envs.franka_env import DefaultEnvConfig
import random
from PIL import Image


class USBEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                self.cap["side_classifier"] = self.cap["side_policy"]
            else:
                cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap

    def reset(self, **kwargs):
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        self._send_gripper_command(1.0)

        # Move above the target pose
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        self.interpolate_move(target, timeout=0.5)
        time.sleep(0.5)
        self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
        time.sleep(0.5)
        self._send_gripper_command(-1.0)

        self._update_currpos()
        reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        reset_pose[1] += 0.04
        self.interpolate_move(reset_pose, timeout=0.5)

        obs, info = super().reset(**kwargs)
        self._send_gripper_command(1.0)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

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

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


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
            print(f"\033[93m Set grasp penalty is {self.penalty}\033[0m")
        else:
            info["grasp_penalty"] = 0.0

        # self.last_gripper_pos = observation["state"][0, 0]
        self.last_gripper_pos = action[-1]
        return observation, reward, terminated, truncated, info


class HumanRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        from pynput import keyboard

        self.success_key = False
        self.failure_key = False
        self.collision_key = False
        self.gripper_coverage_key = False

        def on_press(key):
            try:
                if str(key) == "Key.space":
                    self._set_success()
                elif str(key) == "Key.ctrl_r":
                    self._set_failure()
                elif key.char == ",":
                    self._set_collision()
                elif key.char == ".":
                    self._set_gripper_coverage()
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def _set_success(self):
        # Light Green font
        print("\033[92m Success Key Pressed\033[0m")
        self.success_key = True
    
    def _set_failure(self):
        # light red font
        print("\033[91m Faliure Key Pressed\033[0m")
        self.failure_key = True
    
    def _set_collision(self):
        # light yellow font
        print("\033[93m Collision Key Pressed\033[0m")
        self.collision_key = True
    
    def _set_gripper_coverage(self):
        # light purple font
        print("\033[95m Gripper Coverage Key Pressed\033[0m")
        self.gripper_coverage_key = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if self.success_key:
            reward = 1.0
            # print("\033[92m Reward: 1.0\033[0m")
            self.success_key = False
            done = True
            info["succeed"] = reward
        elif self.failure_key:
            reward = -1.0
            self.failure_key = False
            done = True
            info["succeed"] = reward
        elif self.collision_key:
            reward = -0.2
            self.collision_key = False
            done = False
            info["succeed"] = reward
        elif self.gripper_coverage_key:
            reward = -0.1
            self.gripper_coverage_key = False
            done = False
            info["succeed"] = reward
        else:
            reward = 0.0 if int(reward) == 0 else reward

        return obs, reward, done, truncated, info

    def close(self):
        if hasattr(self, "listener"):
            self.listener.stop()
        return self.env.close()


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError(
                    "If sharpness is a single number, it must be non negative."
                )
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(
                f"{sharpness=} should be a single number or a sequence with length 2."
            )

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(
                f"sharpness values should be between (0., inf), but got {sharpness}."
            )

        return float(sharpness[0]), float(sharpness[1])

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = (
            torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        )
        return {"sharpness_factor": sharpness_factor}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(
            F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor
        )


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(
                f"n_subset should be in the interval [1, {len(transforms)}]"
            )

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class ImageTransformWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        config: DefaultEnvConfig = None
    ):
        super().__init__(env)
        self.tfs = config.TFS
        self.max_num_transforms = config.MAX_NUM_TRANSFORMS
        self.enable_transforms = config.ENABLE_TRANSFORMS
        self.weights = []
        self.transforms = {}
        self.random_order = config.RANDOM_ORDER
        self.camera_sections = config.CAMERA_SECTIONS
        self.p = config.PROBABILITY

        for tf_name, tf_cfg in self.tfs.items():
            if tf_cfg.get("weight", 0.0) <= 0.0:
                continue

            self.transforms[tf_name] = self.make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg["weight"])
        n_subset = min(len(self.transforms), self.max_num_transforms)

        if n_subset == 0 or not self.enable_transforms:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=self.random_order,
            )

    def observation(self, obs):
        for key, value in obs.items():
            if key in self.camera_sections:
                if self.enable_transforms:
                    # change to PIL Image for transform
                    value = value.squeeze(0)
                    img_pil = Image.fromarray(value)
                    if random.random() > self.p:
                        # Add Identity transform helping model to learn the original images
                        tf = v2.Identity()
                        transformed_img_pil = tf(img_pil)
                    else:
                        transformed_img_pil = self.tf(img_pil)
                    obs[key] = np.array(transformed_img_pil)[None]
                else:
                    obs[key] = value
        return obs

    def make_transform_from_config(self, tf_cfg):
        transfrom_type = tf_cfg.get("type", None)
        kwargs = tf_cfg["kwargs"]
        if transfrom_type == "ColorJitter":
            return v2.ColorJitter(**kwargs)
        elif transfrom_type == "SharpnessJitter":
            return SharpnessJitter(**kwargs)
        elif transfrom_type == "RandomAffine":
            return v2.RandomAffine(**kwargs)
        elif transfrom_type == "RandomPerspective":
            return v2.RandomPerspective(**kwargs)
        elif transfrom_type == "RandomResizedCrop":
            return v2.RandomResizedCrop(**kwargs)
        else:
            raise ValueError(f"Transform '{transfrom_type}' is not valid.")
