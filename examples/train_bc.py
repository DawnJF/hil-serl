#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import logging
from serl_launcher.agents.continuous.bc import BCAgent

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig
from utils.tools import setup_logging

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", "usb_pickup_insertion", "Name of experiment corresponding to folder."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string(
    "bc_checkpoint_path", "outputs/bc/debug", "Path to save checkpoints."
)
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 10_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree.leaves

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################


def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            rng, key = jax.random.split(sampling_rng)

            actions = bc_agent.sample_actions(observations=obs, seed=key)
            actions = np.asarray(jax.device_get(actions))
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")


##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        wandb_logger.log({"bc": bc_update_info}, step=step)
        if step % config.log_period == 0 and wandb_logger:
            logging.info(f"bc pretraining step: {step}")
            logging.info(f"bc update info: {bc_update_info}")

        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path),
                bc_agent.state,
                step=step,
                keep=5,
            )
    print_green("bc pretraining done and saved checkpoint")


##############################################################################


def load_demo_data(demo_buffer):
    demo_path = []

    # 定义要加载的文件列表
    data_files = [
        "/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-09/usb_pickup_insertion_30_11-50-21.pkl",
        # classifier_data 子目录中的文件
        "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-12/*.pkl",
        "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-12-13/*.pkl",
        "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-15/*.pkl",
    ]

    import glob

    for file_pattern in data_files:
        # 处理通配符模式
        if "*" in file_pattern:
            # 使用 glob 查找匹配的文件

            matched_files = glob.glob(file_pattern)

            if not matched_files:
                logging.warning(f"没有找到匹配的文件: {file_pattern}")
                continue
            for file_path in matched_files:
                demo_path.append(file_path)
        else:
            demo_path.append(file_pattern)

    for path in demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                if np.linalg.norm(transition["actions"]) > 0.0:
                    demo_buffer.insert(transition)


def main(_):

    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    bc_agent: BCAgent = jax.device_put(
        jax.tree_map(jnp.array, bc_agent), sharding.replicate()
    )

    if not eval_mode:
        FLAGS.bc_checkpoint_path = os.path.join(
            FLAGS.bc_checkpoint_path, time.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(FLAGS.bc_checkpoint_path, exist_ok=True)
        assert not os.path.isdir(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        setup_logging(FLAGS.bc_checkpoint_path)

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        load_demo_data(bc_replay_buffer)
        print_green(f"bc replay buffer size: {len(bc_replay_buffer)}")

        # train loop
        print_green("starting train loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

        bc_ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("starting actor loop")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
