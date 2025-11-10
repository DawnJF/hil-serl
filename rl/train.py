#!/usr/bin/env python3

import glob
import time
import jax

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree.leaves
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from natsort import natsorted
import sys

sys.path.append(os.getcwd())
from rl.envs_store import RecordEpisodeStatistics
from rl.sac_hybrid_single import SACAgentHybridSingleArm, HybridSACAgent
from serl_launcher.serl_launcher.utils.timer_utils import Timer
from serl_launcher.serl_launcher.utils.train_utils import concat_batches

# from serl_launcher.serl_launcher.agents.continuous.bc import BCAgent
# from serl_launcher.serl_launcher.utils.launcher import make_bc_agent

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from rl.launcher import (
    make_sac_pixel_agent_hybrid_single_arm,
    make_hybrid_sac_agent,
    make_trainer_config,
    make_wandb_logger,
)
from rl.buffer_tools import MemoryEfficientReplayBufferDataStore
from rl.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name",
    "plug_into_socket_with_power_cord",
    "Name of experiment config",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("port", 5188, "port")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string(
    "checkpoint_path",
    "outputs/rlpd/plug_into_socket_with_power_cord_dqn",
    "Path to save checkpoints.",
)
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save BC checkpoints for IBRL")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("training_starts", 100, "Wait")
flags.DEFINE_integer("target_entropy", -4, "target_entropy")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string(
    "wandb_mode",
    "online",
    "wandb mode, online or offline, if debug is true, mode is disabled.",
)
flags.DEFINE_string("wandb_output_dir", None, "wandb output dir")

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


def select_action_v2(actions, bc_agent, obs, agent, seed):
    if bc_agent is None:
        return actions

    xyz = actions[:3]
    bc_actions = bc_agent.sample_actions(
        observations=jax.device_put(obs),
        seed=seed,
        argmax=True,
    )

    bc_xyz = bc_actions[:3]

    q = agent.forward_critic_eval(obs, xyz)
    bc_q = agent.forward_critic_eval(obs, bc_xyz)

    if bc_q.min(axis=0) > q.min(axis=0):
        return bc_actions
    else:
        return actions


def load_demo_data(demo_buffer):
    # assert FLAGS.demo_path is not None
    if FLAGS.demo_path is None:
        FLAGS.demo_path = []
        print_green("No demo path provided, proceeding with empty demo buffer.")
    for path in FLAGS.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                if "infos" in transition and "grasp_penalty" in transition["infos"]:
                    transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                demo_buffer.insert(transition)


def resume_data_from_checkpoint(replay_buffer, demo_buffer):
    if FLAGS.checkpoint_path is not None and os.path.exists(
        os.path.join(FLAGS.checkpoint_path, "buffer")
    ):
        for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    replay_buffer.insert(transition)
        print_green(
            f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
        )

    if FLAGS.checkpoint_path is not None and os.path.exists(
        os.path.join(FLAGS.checkpoint_path, "demo_buffer")
    ):
        for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    demo_buffer.insert(transition)
        print_green(
            f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
        )


##############################################################################


def actor(
    agent,
    data_store,
    intvn_data_store,
    env,
    sampling_rng,
    bc_agent=None,
    include_o_actions=False,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    start_step = 0

    if FLAGS.checkpoint_path and os.path.exists(
        os.path.join(FLAGS.checkpoint_path, "buffer")
    ):
        resume_buffer_list = glob.glob(
            os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")
        )
        if len(resume_buffer_list) > 0:
            start_step = int(os.path.basename(natsorted(resume_buffer_list)[-1])[12:-4])
        +1

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port, broadcast_port=FLAGS.port + 1),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )

                actions = select_action_v2(actions, bc_agent, obs, agent, key)
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            o_actions = actions.copy()  # Always save original action
            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if "grasp_penalty" in info:
                transition["grasp_penalty"] = info["grasp_penalty"]
            if include_o_actions:
                transition["o_actions"] = o_actions
                transition["h"] = already_intervened
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    resume_cp = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
    start_step = (
        int(os.path.basename(resume_cp)[11:]) + 1 if resume_cp is not None else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""

        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)

        return {}  # not expecting a response

    # Create server
    server = TrainerServer(
        make_trainer_config(port_number=FLAGS.port, broadcast_port=FLAGS.port + 1),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    print_green(f"waiting for demo buffer {len(demo_buffer)} / {FLAGS.training_starts}")
    while len(demo_buffer) < FLAGS.training_starts:
        time.sleep(1)

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    if isinstance(agent, HybridSACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset(
            {"critic", "grasp_critic", "actor", "temperature"}
        )

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)

            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    bc_agent = None
    # if FLAGS.bc_checkpoint_path is not None:
    #     bc_agent: BCAgent = make_bc_agent(
    #         seed=FLAGS.seed,
    #         sample_obs=env.observation_space.sample(),
    #         sample_action=env.action_space.sample(),
    #         image_keys=config.image_keys,
    #         encoder_type=config.encoder_type,
    #     )
    #     bc_agent: BCAgent = jax.device_put(
    #         jax.tree_map(jnp.array, bc_agent), sharding.replicate()
    #     )
    #     bc_ckpt = checkpoints.restore_checkpoint(
    #         FLAGS.bc_checkpoint_path,
    #         bc_agent.state,
    #     )
    #     bc_agent = bc_agent.replace(state=bc_ckpt)

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
        target_entropy=FLAGS.target_entropy,
    )
    include_grasp_penalty = True
    include_o_actions = False  # 还没测试通过

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input(
            f"Checkpoint path {FLAGS.checkpoint_path} already exists. Press Enter to resume training."
        )
        ckpt_path = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
        print_green(f"Resuming from checkpoint: {ckpt_path}")
        if ckpt_path is not None:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(
                checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
            )[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_o_actions=include_o_actions,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
            mode=FLAGS.wandb_mode,
            output_dir=FLAGS.wandb_output_dir,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_o_actions=include_o_actions,
        )

        load_demo_data(demo_buffer)
        resume_data_from_checkpoint(replay_buffer, demo_buffer)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            bc_agent=bc_agent,
            include_o_actions=include_o_actions,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
