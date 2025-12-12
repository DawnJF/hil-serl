import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import jax
from typing import Union
from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import make_sac_cql_pixel_agent_hybrid_single_arm, make_calql_pixel_agent
from serl_launcher.agents.continuous.cql_hybrid_single import CQLAgentHybridSingleArm
from flax.training import checkpoints
from serl_launcher.agents.continuous.calql import CalQLAgent
from serl_launcher.agents.continuous.cql import CQLAgent
import jax.numpy as jnp


if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree.leaves


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", "usb_pickup_insertion", "Name of experiment corresponding to folder."
)
flags.DEFINE_integer("successes_needed", 30, "Number of successful demos to collect.")
flags.DEFINE_boolean("collect_data_wsrl_way", False, "Collect data by wsrl way")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string(
    "pretrained_checkpoint_path",
    None,
    "Path to the pre-trained offline RL agent checkpoint.",
)


def make_agent(config, env):
    if config.setup_mode == 'calql':
        agent: Union[CalQLAgent, CQLAgent] = make_calql_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            reward_scale=1.0,
            reward_bias=0.0,  # eventually move this to the env?
            discount=config.discount,
            is_calql=True,
            target_entropy=-4.0
        )
    elif config.setup_mode == 'calql-single-arm-learned-gripper':
        agent: CQLAgentHybridSingleArm = make_sac_cql_pixel_agent_hybrid_single_arm(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type,
                reward_scale=1.0,
                reward_bias=0.0,
                discount=config.discount,
            )
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.pretrained_checkpoint_path),
        agent.state,
    )
    agent = agent.replace(state=ckpt)
    return agent

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    if FLAGS.collect_data_wsrl_way:
        agent = make_agent(config, env)
        sampling_rng = jax.random.PRNGKey(FLAGS.seed)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    # detect the KeyboardInterrupt
    try:
        while success_count < success_needed:            
            if FLAGS.collect_data_wsrl_way:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = actions.at[-1].set(0.0) if actions.shape[-1] == 4 else jnp.concatenate([actions, jnp.zeros((1,))], axis=-1)
            else:
                actions = np.zeros(env.action_space.sample().shape)

            next_obs, rew, done, truncated, info = env.step(actions)
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]

            if (
                not done
                and not rew
                and actions[0] == 0
                and actions[1] == 0
                and actions[2] == 0
                and actions[3] == 0
            ):
                # print("Note: not moving!")
                continue

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)

            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                print(f"\033[33m Done: {done}, Reward: {rew}, info.succeed: {info['succeed']}\033[0m")
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                # After reset, we should suspend 5s for reposition object.
                print("\033[31m WILL TO BE RESETED\033[0m")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print(f"\nDetect Ctrl+C，save collected {success_count} demo data...")
        success_needed = success_count

    # if not os.path.exists("./demo_data"):
    #     os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ymd = uuid.split("_")[0]
    hms = uuid.split("_")[1]
    file_dir = f"/home/facelesswei/code/Jax_Hil_Serl_Dataset/{ymd}"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"/home/facelesswei/code/Jax_Hil_Serl_Dataset/{ymd}/{FLAGS.exp_name}_{success_needed}_{hms}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")
    env.expert.close()
    env.close()


if __name__ == "__main__":
    app.run(main)
