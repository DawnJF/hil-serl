import os
import sys
import numpy as np
import torch


sys.path.append(os.getcwd())
from rl.envs_temp import get_environment
from rl.sac_policy import SACConfig, SACPolicy, dict_data_to_torch, get_eval_transform


def main():
    cp = "outputs/torch_rlpd/debug/20251014-1726/checkpoint_10000.pt"
    env = get_environment()

    sac_config = SACConfig()
    agent = SACPolicy(sac_config)

    device = torch.device("cuda:0")
    agent.prepare(device)

    agent.load_checkpoint(cp)
    agent.eval()

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
