import os
import sys
import numpy as np
import torch


sys.path.append(os.getcwd())
from rl.envs_temp import get_environment
from rl.sac_policy import SACConfig, SACPolicy, dict_data_to_torch, get_eval_transform


def main():
    env = get_environment()

    sac_config = SACConfig()
    agent = SACPolicy(sac_config)
    agent.eval()

    device = torch.device("cuda:0")
    agent.prepare(device)

    obs, info = env.reset()

    for step in range(1000):

        obs_torch = dict_data_to_torch(obs, get_eval_transform(), device=device)
        actions = agent.sample_actions(obs_torch, argmax=True)

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions).squeeze()

        obs, reward, done, truncated, info = env.step(actions)

        if done:
            obs, info = env.reset()


if __name__ == "__main__":
    main()
