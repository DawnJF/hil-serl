#!/usr/bin/env python3

import time
import jax

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree.leaves
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints
import os
import sys

sys.path.append(os.getcwd())
from rl.envs_store import RecordEpisodeStatistics
from rl.sac_hybrid_single import SACAgentHybridSingleArm, HybridSACAgent
from rl.launcher import make_sac_pixel_agent_hybrid_single_arm
from rl.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")


# for optimizer config
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_integer("warmup_steps", 0, "warm-up steps")
flags.DEFINE_integer("cosine_decay_steps", None, "cosing decay steps")
flags.DEFINE_float("weight_decay", None, "weight decay for adamw")
flags.DEFINE_float("clip_grad_norm", None, "clip grad norm intensity")
flags.DEFINE_boolean("return_lr_schedule", False, "if return lr schedule")


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def inference(agent, env, sampling_rng):
    success_counter = 0
    time_list = []

    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path),
        agent.state,
        step=FLAGS.eval_checkpoint_step,
    )
    agent = agent.replace(state=ckpt)

    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs), argmax=True, seed=key
            )
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
    return  # after done eval, return and exit


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(debug=True)
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    env.reset()

    agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
        # agent: HybridSACAgent = make_hybrid_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        discount=config.discount,
        optimizer_configs={
            "learning_rate": FLAGS.learning_rate,
            "warmup_steps": FLAGS.warmup_steps,
            "cosine_decay_steps": FLAGS.cosine_decay_steps,
            "weight_decay": FLAGS.weight_decay,
            "clip_grad_norm": FLAGS.clip_grad_norm,
            "return_lr_schedule": FLAGS.return_lr_schedule,
        },
    )
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

    # actor loop
    print_green("starting actor loop")
    inference(
        agent,
        env,
        sampling_rng,
    )


if __name__ == "__main__":
    app.run(main)
