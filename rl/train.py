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
from torch.utils.tensorboard import SummaryWriter
from agentlace.trainer import TrainerServer, TrainerClient, TrainerConfig
from agentlace.data.data_store import QueuedDataStore

sys.path.append(os.getcwd())
from bc.train_bc2rl import RLActor
from serl_launcher.serl_launcher.utils.logging_utils import RecordEpisodeStatistics
from serl_launcher.serl_launcher.utils.timer_utils import Timer
from rl.envs_temp import get_environment, get_fake_environment
from rl.replay_buffer_data_store import ReplayBufferDataStore
from rl.sac_policy import (
    SACPolicy,
    SACConfig,
    dict_data_to_torch,
    get_train_transform,
    get_eval_transform,
)
from utils.tools import get_device, print_dict_structure, print_dict_device


@dataclass
class Config:
    exp_name: str = "usb_pickup_insertion"
    seed: int = 42
    learner: bool = False
    actor: bool = False
    ip: str = "localhost"
    port: int = 5588
    demo_path: Optional[List[str]] = field(default=None)
    checkpoint_path: str = "outputs/torch_rlpd/debug"
    resume_checkpoint: Optional[str] = None  # 恢复训练的checkpoint路径
    resume_actor: Optional[str] = None
    freeze_actor: bool = False
    debug: bool = False

    # Training parameters
    max_steps: int = 1000000
    random_steps: int = 0
    batch_size: int = 256  # 256
    training_starts: int = 150
    replay_buffer_capacity: int = 200000
    cta_ratio: int = 2  # critic to actor update ratio
    steps_per_update: int = 50  # steps between network updates
    checkpoint_period: int = 400
    buffer_period: int = 2000  # steps between buffer saves
    image_keys: List[str] = field(default_factory=lambda: ["rgb", "wrist"])


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats", "request-q"],
    )


def make_tensorboard_logger(log_dir, debug=False):
    """Create tensorboard SummaryWriter"""
    writer = SummaryWriter(log_dir)

    class TensorboardLogger:
        def __init__(self, writer, debug=False):
            self.writer = writer
            self.debug = debug

        def log(self, data, step=None):
            if self.debug:
                print(f"Step {step}")
                print_dict_structure(data)

            if step is not None:
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float, np.number)):
                                self.writer.add_scalar(
                                    f"{key}/{sub_key}", sub_value, step
                                )
                    elif isinstance(value, (int, float, np.number)):
                        self.writer.add_scalar(key, value, step)

        def close(self):
            self.writer.close()

    return TensorboardLogger(writer, debug)


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
            raise ValueError("Unsupported batch data type")


def prefetch_batch_async(iterator, device, image_transform, num_prefetch=2):
    """异步数据预取函数"""
    import queue
    import threading

    batch_queue = queue.Queue(maxsize=num_prefetch)

    def worker():
        try:
            while True:
                replay_batch = next(iterator[0])
                demo_batch = next(iterator[1])
                batch = concat_batches(replay_batch, demo_batch, axis=0)

                # 在后台线程中转换数据
                from rl.sac_policy import dict_data_to_torch

                batch = dict_data_to_torch(batch, image_transform, device=device)

                batch_queue.put(batch)
        except StopIteration:
            batch_queue.put(None)

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

    while True:
        batch = batch_queue.get()
        if batch is None:
            break
        yield batch


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


def actor(
    config, agent: SACPolicy, data_store, intvn_data_store, env, device, bc_agent=None
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    start_step = 0

    agent.eval()

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
        # 使用新的load_params方法更新参数
        print_green("update_params")
        agent.load_params(params)

    client.recv_network_callback(update_params)
    print_green("connected to server")

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

    eval_image_transform = get_eval_transform()

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                obs_torch = dict_data_to_torch(obs, eval_image_transform, device=device)
                actions = agent.sample_actions(obs_torch, argmax=False)

                actions = select_action_v2(actions, bc_agent, obs, agent)

                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                actions = np.asarray(actions).squeeze()

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
                print(f"Episode done at step {step}, return: {running_return}")

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

            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % 10 == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(
    config: Config,
    agent: SACPolicy,
    replay_buffer,
    demo_buffer,
    tb_logger,
    device,
    start_step: int = 0,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    step = start_step

    if config.resume_actor is not None:
        c_dict, d_dict = RLActor.READ_CHECKPOINT(config.resume_actor)
        agent.actor.load_state_dict(c_dict)
        agent.discrete_critic.load_state_dict(d_dict)
        print_green(f"resume_actor from {config.resume_actor}")

    if config.freeze_actor:
        agent.freeze_bc_actor()
        print_green("Froze the actor and discrete critic parameters.")

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""

        if type == "send-stats":
            if tb_logger is not None:
                tb_logger.log(payload, step=step)
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

    time.sleep(3)  # wait for client to connect

    # send the initial network to the actor
    server.publish_network(agent.get_params())
    print_green("sent initial network to actor")

    # Loop to wait until replay_buffer is filled
    print_green(
        f"waiting for replay buffer {len(replay_buffer)} / {config.training_starts}"
    )
    while len(replay_buffer) < config.training_starts:
        time.sleep(1)

    print_green(
        f"waiting for demo buffer {len(demo_buffer)} / {config.training_starts}"
    )
    while len(demo_buffer) < config.training_starts:
        time.sleep(1)

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

    timer = Timer()

    train_image_transform = get_train_transform()

    print_green("starting learner loop")
    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                replay_batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(replay_batch, demo_batch, axis=0)
                batch = dict_data_to_torch(batch, train_image_transform, device=device)

            with timer.context("train_critics"):
                update_info = agent.train_step(batch, critic_only=True)

        with timer.context("sample_replay_buffer"):
            replay_batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(replay_batch, demo_batch, axis=0)
            batch = dict_data_to_torch(batch, train_image_transform, device=device)

        with timer.context("train"):
            update_info = agent.train_step(batch, critic_only=False)

        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            with timer.context("publish_network"):
                server.publish_network(agent.get_params())

        if tb_logger:
            tb_logger.log(update_info, step=step)
            tb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and step % config.checkpoint_period == 0:
            # 保存checkpoint

            checkpoint_file = os.path.join(
                config.checkpoint_path, f"checkpoint_{step}.pt"
            )

            # 保存完整的训练状态
            additional_info = {
                "replay_buffer_size": len(replay_buffer),
                "demo_buffer_size": len(demo_buffer),
                "config": config,
                "timer_stats": timer.get_average_times() if "timer" in locals() else {},
            }
            agent.save_checkpoint(
                checkpoint_file, step=step, additional_info=additional_info
            )
            print_green(f"Saved checkpoint at step {step}: {checkpoint_file}")


##############################################################################


def load_demo_data(config, demo_buffer):
    if config.demo_path is None:
        return
    print_green(f"Loading demo data from: {config.demo_path}")
    for path in config.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                if "infos" in transition and "grasp_penalty" in transition["infos"]:
                    transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                demo_buffer.insert(transition)


def resume_buffer_from_checkpoint(config: Config, replay_buffer, demo_buffer):
    if config.resume_checkpoint is None:
        return
    folder = os.path.dirname(config.resume_checkpoint)
    if os.path.exists(os.path.join(folder, "buffer")):
        for file in glob.glob(os.path.join(folder, "buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    replay_buffer.insert(transition)
        print_green(
            f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
        )

    if os.path.exists(os.path.join(folder, "demo_buffer")):
        for file in glob.glob(os.path.join(folder, "demo_buffer/*.pkl")):
            with open(file, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    demo_buffer.insert(transition)
        print_green(
            f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
        )


def fake_bc_agent(obs):
    # 返回固定的动作，可以根据需要调整
    return np.zeros(3)  # 只返回3维动作


def main(config: Config):
    config.checkpoint_path = os.path.join(
        config.checkpoint_path, time.strftime("%Y%m%d-%H%M")
    )
    os.makedirs(config.checkpoint_path, exist_ok=True)
    print_green(f"Experiment outputs will be saved to: {config.checkpoint_path}")

    torch.backends.cudnn.benchmark = True  # 提升卷积性能

    # env = get_fake_environment()
    env = get_environment()
    env = RecordEpisodeStatistics(env)

    # 初始化SAC agent
    sac_config = SACConfig()
    agent = SACPolicy(sac_config)

    include_grasp_penalty = True

    bc_agent = None
    # bc_agent = fake_bc_agent

    # 检查是否需要从checkpoint恢复训练
    resume_step = 0

    if config.learner:
        device = torch.device("cuda:1")
        agent.prepare(device)
        print_green(f"Moved SAC agent to {device}")

        if config.resume_checkpoint:
            checkpoint_metadata = agent.load_checkpoint(config.resume_checkpoint)

            resume_step = checkpoint_metadata.get("step", 0)
            print_green(
                f"Resumed training from step: {resume_step} : {config.resume_checkpoint}"
            )

        # set up tensorboard logging
        tb_logger = make_tensorboard_logger(
            log_dir=config.checkpoint_path,
            debug=config.debug,
        )

        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            include_grasp_penalty=include_grasp_penalty,
        )
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            include_grasp_penalty=include_grasp_penalty,
        )

        load_demo_data(config, demo_buffer)
        resume_buffer_from_checkpoint(config, replay_buffer, demo_buffer)

        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        # learner loop
        learner(
            config,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            tb_logger=tb_logger,
            device=device,
            start_step=resume_step,
        )

    elif config.actor:
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)  # demo_buffer

        device = torch.device("cuda:0")
        agent.prepare(device)
        print_green(f"Moved SAC agent to {device}")

        # actor loop
        print_green("starting actor loop")
        actor(config, agent, data_store, intvn_data_store, env, device, bc_agent)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)


"""
# 从头开始训练
python rl/train.py --learner --demo_path "/Users/majianfei/Downloads/usb_pickup_insertion_5_11-05-02.pkl" --max_steps 100

python rl/train.py --actor --debug --max_steps 100

# 从checkpoint恢复训练
python rl/train.py --learner --resume_checkpoint "outputs/torch_rlpd/debug/20250930-1441/checkpoint_400.pt"


python rl/train.py --learner --freeze_actor --resume_actor outputs/bc2rl_20250928_173221/checkpoint-32.pth 
"""


"""

train: 24
train_critics: 16
sample_replay_buffer: 18


"""
