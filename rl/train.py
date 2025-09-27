import sys
import os
import glob
import time
import numpy as np
import torch
import tqdm
import tyro
from dataclasses import dataclass, field
from typing import Optional, List
import copy
import pickle as pkl
from agentlace.trainer import TrainerServer, TrainerClient, TrainerConfig
from agentlace.data.data_store import QueuedDataStore

sys.path.append(os.getcwd())
from serl_launcher.serl_launcher.utils.logging_utils import RecordEpisodeStatistics
from serl_launcher.serl_launcher.utils.timer_utils import Timer
from rl.envs_temp import get_environment
from rl.replay_buffer_data_store import ReplayBufferDataStore
from rl.sac_policy import SACPolicy, SACConfig, dict_data_to_torch, get_train_transform
from utils.tools import get_device, print_dict_structure


@dataclass
class Config:
    exp_name: str = "usb_pickup_insertion"
    seed: int = 42
    learner: bool = False
    actor: bool = False
    ip: str = "localhost"
    port: int = 5588
    demo_path: Optional[List[str]] = field(default=None)
    checkpoint_path: str = "outputs/debug"
    eval_n_trajs: int = 0
    save_video: bool = False
    debug: bool = False
    wandb_mode: str = "offline"
    wandb_output_dir: Optional[str] = None

    # Training parameters
    max_steps: int = 1000000
    random_steps: int = 2000
    batch_size: int = 512
    training_starts: int = 2000
    replay_buffer_capacity: int = 1000000
    cta_ratio: int = 1  # critic to actor update ratio
    steps_per_update: int = 200  # steps between network updates
    log_period: int = 1000
    checkpoint_period: int = 10000
    buffer_period: int = 25000  # steps between buffer saves
    image_keys: List[str] = field(default_factory=lambda: ["rgb", "wrist"])


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats", "request-q"],
    )


def make_wandb_logger(
    project, description, debug=False, mode="offline", output_dir=None
):
    """Simple wandb logger placeholder"""

    class DummyLogger:
        def log(self, data, step=None):
            if debug:
                print(f"Step {step}: {data}")

    return DummyLogger()


device = get_device()
print(f"Using device: {device}")


# 简化版本的 concat_batches 函数
def concat_batches(batch1, batch2, axis=0):
    """Concatenate two batches along specified axis"""
    if isinstance(batch1, dict):
        return {k: concat_batches(batch1[k], batch2[k], axis) for k in batch1.keys()}
    else:
        if hasattr(batch1, "shape"):  # numpy array or tensor
            if isinstance(batch1, torch.Tensor):
                return torch.cat([batch1, batch2], dim=axis)
            else:
                return np.concatenate([batch1, batch2], axis=axis)
        else:
            return batch1


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def select_action_v2(actions, bc_agent, obs, agent):
    if bc_agent is None:
        return actions

    xyz = actions[:3]
    bc_actions = bc_agent(obs)
    if isinstance(bc_actions, (list, tuple)):
        bc_actions = bc_actions[0]
    if isinstance(bc_actions, torch.Tensor):
        bc_actions = torch.cat([bc_actions, torch.tensor([actions[3]])], dim=0)
    else:
        bc_actions = np.append(bc_actions, actions[3])

    bc_xyz = bc_actions[:3]

    # 转换为适合的格式调用forward_critic_eval
    obs_for_critic = {"state": obs} if not isinstance(obs, dict) else obs

    with torch.no_grad():
        q = agent.forward_critic_eval(obs_for_critic, xyz)
        bc_q = agent.forward_critic_eval(obs_for_critic, bc_xyz)

        if isinstance(q, torch.Tensor):
            q_min = q.min()
            bc_q_min = bc_q.min()
        else:
            q_min = q.min(axis=0)
            bc_q_min = bc_q.min(axis=0)

        if bc_q_min > q_min:
            return bc_actions
        else:
            return actions


##############################################################################


def actor(config, agent, data_store, intvn_data_store, env, bc_agent=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    start_step = 0

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        config.ip,
        make_trainer_config(port_number=config.port, broadcast_port=config.port + 1),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        # PyTorch风格的参数更新
        for name, param in agent.named_parameters():
            if name in params:
                param.data.copy_(params[name])

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
                actions = agent.sample_actions(
                    obs,
                    argmax=False,
                )
                # actions = select_action(actions, bc_agent, obs, client)
                actions = select_action_v2(actions, bc_agent, obs, agent)
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                actions = np.asarray(actions)

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

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
            buffer_path = os.path.join(config.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(config.checkpoint_path, "demo_buffer")
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


def learner(
    config: Config, agent: SACPolicy, replay_buffer, demo_buffer, wandb_logger=None
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = 0  # 简化版本，暂时不支持从checkpoint恢复
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""

        if type == "send-stats":
            if wandb_logger is not None:
                wandb_logger.log(payload, step=step)
        else:
            raise ValueError(f"Invalid request type: {type}")

        return {}  # not expecting a response

    # Create server
    server = TrainerServer(
        make_trainer_config(port_number=config.port, broadcast_port=config.port + 1),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(
        {name: param.data.clone() for name, param in agent.named_parameters()}
    )
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

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
                batch = dict_data_to_torch(batch, get_train_transform())
                update_info = agent.train_step(batch, critic_only=True)

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            batch = dict_data_to_torch(batch, get_train_transform())
            update_info = agent.train_step(batch, critic_only=False)
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            server.publish_network(
                {name: param.data.clone() for name, param in agent.named_parameters()}
            )

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and step % config.checkpoint_period == 0:
            pass

    ##############################################################################


def create_replay_buffer_and_wandb_logger(config: Config, env, include_grasp_penalty):
    replay_buffer = ReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=config.replay_buffer_capacity,
        include_grasp_penalty=include_grasp_penalty,
    )
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="hil-serl",
        description=config.exp_name,
        debug=config.debug,
        mode=config.wandb_mode,
        output_dir=config.wandb_output_dir,
    )
    return replay_buffer, wandb_logger


def main(config: Config):
    env = get_environment()
    env = RecordEpisodeStatistics(env)

    # 初始化SAC agent
    sac_config = SACConfig()
    agent = SACPolicy(sac_config)
    include_grasp_penalty = True

    # 简化版本的 bc_agent
    bc_agent = None

    def fake_bc_agent(obs):
        # 返回固定的动作，可以根据需要调整
        return np.zeros(3)  # 只返回3维动作

    bc_agent = fake_bc_agent

    # TODO resume training

    if config.learner:

        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger(
            config, env, include_grasp_penalty
        )
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert config.demo_path is not None
        for path in config.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if "infos" in transition and "grasp_penalty" in transition["infos"]:
                        transition["grasp_penalty"] = transition["infos"][
                            "grasp_penalty"
                        ]
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if config.checkpoint_path is not None and os.path.exists(
            os.path.join(config.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(config.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if config.checkpoint_path is not None and os.path.exists(
            os.path.join(config.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(config.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            config,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif config.actor:
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(config, agent, data_store, intvn_data_store, env, bc_agent)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)


"""
python rl/train.py --learner --demo_path "/Users/majianfei/Downloads/usb_pickup_insertion_5_11-05-02.pkl" --debug --max_steps 100

python rl/train.py --actor --debug --max_steps 100
"""
