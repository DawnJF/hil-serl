import time
from typing import Callable
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from gymnasium import Env, spaces


class RewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, multi=True, model_path=None):
        super().__init__(env)

        self.classfier_model_path = model_path
        self.multi = multi  # 先设置 multi 属性

        self.reward_classifier_func = RewardModelInferencer(
            model_path=self.classfier_model_path, multi=self.multi  # 现在可以使用了
        )

    def compute_reward(self, obs):
        if self.classfier_model_path is None:
            raise ValueError("Please set the model_path for the reward classifier")
        return self.reward_classifier_func.predict(obs)

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info["succeed"] = bool(rew)

        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet18", pretrained=True):
        super().__init__()
        # 选择 backbone
        if backbone == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)

        # 替换掉最后的分类层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 512)

        # 添加额外的层来处理两张图片的特征
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 两张图片的特征拼接
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x1, x2):
        """
        x: Tensor, shape [B, N, C, H, W]
        返回: [B, num_classes] - 对序列进行平均池化
        """

        def _transform(x):
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)  # 合并 batch 和 N
            return x

        x1 = _transform(x1)
        x2 = _transform(x2)
        # 提取第一张图片的特征
        feat1 = self.model(x1)  # [B, 512]

        # 提取第二张图片的特征
        feat2 = self.model(x2)  # [B, 512]

        # 拼接两张图片的特征
        combined = torch.cat([feat1, feat2], dim=1)  # [B, 1024]
        # 分类
        out = self.classifier(combined)  # [B, num_classes]

        return out


class RewardModelInferencer:
    def __init__(self, model_path, device=None, multi=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNetClassifier().to(self.device)
        load_checkpoint(self.model, path=model_path)
        self.model.eval()
        self.multi = multi
        self.num = 0

    def predict(self, obs):
        if self.multi:
            return self.multi_predict(obs["images"]["rgb"], obs["images"]["wrist"])
        else:
            return self.single_predict(obs["images"]["rgb"])

    def single_predict(self, np_img):
        """
        np_imgs: numpy.ndarray, shape [N, C, H, W] or list of np.ndarray [C, H, W]
        返回: int, 预测类别
        """
        # print(f"np_img: {np_img.shape}")
        if np_img.shape[0] != 3:
            np_img = np_img.transpose(2, 0, 1)
        #     imgs = [np_img]
        # imgs = torch.stack(imgs, dim=0)  # [N, C, H, W]
        np_img = image_normalization(np_img)
        imgs = (
            torch.tensor(np_img, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            # print(imgs.shape)
            outputs = self.model(imgs)  # [1, num_classes]
            probs = torch.softmax(outputs, dim=1)  # 转成概率
            p1 = probs[0, 1].item()  # 类别 1 的概率

            threshold = 0.7

            print(f"predict: {p1:.4f}: {p1 > threshold}")
            return 1 if p1 > threshold else 0  # reward

    def multi_predict(self, np_rgb, np_wrist):
        """
        np_imgs: numpy.ndarray, shape [N, C, H, W] or list of np.ndarray [C, H, W]
        返回: int, 预测类别
        """

        def _transform(img, crop_part):

            img = image_normalization(img)
            img = (
                torch.tensor(img, dtype=torch.float32)
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.num += 1
            return img

        np_rgb = _transform(np_rgb, "rgb")
        np_wrist = _transform(np_wrist, "wrist")

        with torch.no_grad():
            # print(imgs.shape)
            outputs = self.model(np_rgb, np_wrist)  # [1, num_classes]
            print(f"outputs: {outputs}")
            # outputs: tensor([[ 2.2152, -1.8834]], device='cuda:0')
            probs = torch.softmax(outputs, dim=1)  # 转成概率
            print(f"probs: {probs}")
            p1 = probs[0, 1].item()  # 类别 1 的概率

            threshold = 0.90

            print(f"predict: {p1:.4f}: {p1 > threshold}")
            return 1 if p1 > threshold else 0


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {path}")


def image_normalization(img):
    # 如果图像是 HWC 格式，转换为 CHW 格式
    if img.shape[-1] == 3:  # 最后一个维度是通道数
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    normalized_img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    standardized_img = (normalized_img - mean) / std
    return standardized_img


if __name__ == "__main__":

    import os
    import sys

    sys.path.append(os.getcwd())
    sys.path.append("/home/facelesswei/code/hil-serl")
    sys.path.append("/home/facelesswei/code/hil-serl/examples")
    from examples.experiments.usb_pickup_insertion.config import UREnvConfig
    from franka_env.envs.wrappers import SpacemouseIntervention
    from examples.experiments.usb_pickup_insertion.ur_wrapper import UR_Platform_Env

    # 创建原始环境
    env = UR_Platform_Env(fake_env=False, config=UREnvConfig())
    # env = SpacemouseIntervention(env)
    # 包装成带奖励分类器的环境
    env = RewardClassifierWrapper(
        env=env,
        multi=True,
        model_path="/home/facelesswei/code/debug_UR_Robot_Arm_Show/reward_model/checkpoint_plug_multi/checkpoint-1.pth",
    )
    action = env.action_space.sample()
    action = np.zeros((7,))
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward}")
    # reward 现在是由分类器计算得出的
    # info 包含 'env_reward' 和 'classifier_reward'
