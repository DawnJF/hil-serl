#!/usr/bin/env python3


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
import pickle as pkl

from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm

from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name1", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("exp_name2", None, "Name of experiment corresponding to folder.")
flags.DEFINE_string("exp_name3", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path1", None, "Path to save checkpoints.")
flags.DEFINE_string("checkpoint_path2", None, "Path to save checkpoints.")
flags.DEFINE_string("checkpoint_path3", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step1", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_checkpoint_step2", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_checkpoint_step3", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string(
    "wandb_mode",
    "online",
    "wandb mode, online or offline, if debug is true, mode is disabled.",
)

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


def inference(configs, paths, steps, sampling_rng):

    for config, ckpt_path, ckpt_step in zip(configs, paths, steps):

        env = config.get_environment(debug=True)

        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
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

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(ckpt_path),
            agent.state,
            step=ckpt_step,
        )
        agent = agent.replace(state=ckpt)

        done = False

        obs, _ = env.reset()
        # obs, reward, _, truncated, info = env.step(np.zeros_like(env.action_space.sample()))
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs), argmax=True, seed=key
            )
            actions = np.asarray(jax.device_get(actions))

            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
        # env.expert.close()
        env.close()

        print("=========end==========")


def replay(configs, paths):

    for config, path in zip(configs, paths):

        env = config.get_environment()

        obs, _ = env.reset()

        with open(path, "rb") as f:
            transitions = pkl.load(f)
            print(transitions[0].keys())

            for i in range(len(transitions)):
                print("==== step ", i, " ====")
                print(f"reward: {transitions[i]['rewards']}")
                print(f"done: {transitions[i]['dones']}")
                print(f"action: {transitions[i]['actions']}")
                print(f"masks: {transitions[i]['masks']}")

                env.step(transitions[i]["actions"])
                if transitions[i]["dones"]:
                    print(f"Episode {i} finished.")
                    break

        env.close()
    return  # after done eval, return and exit


def main(_):
    global config1, config2, config3
    config1 = CONFIG_MAPPING[FLAGS.exp_name1]()
    config2 = CONFIG_MAPPING[FLAGS.exp_name2]()
    config3 = CONFIG_MAPPING[FLAGS.exp_name3]()
    configs = [config1, config2]
    paths = [FLAGS.checkpoint_path1, FLAGS.checkpoint_path2]
    steps = [FLAGS.eval_checkpoint_step1, FLAGS.eval_checkpoint_step2]

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    sampling_rng = jax.device_put(sampling_rng, sharding.replicate())

    print_green("starting inference")
    inference(
        configs,
        paths,
        steps,
        sampling_rng,
    )
    # replay([config3], [FLAGS.checkpoint_path3])


if __name__ == "__main__":
    app.run(main)
