import copy
from functools import partial
from typing import Optional, Iterable

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from ml_collections import ConfigDict

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ValueCritic, ensemblize
from serl_launcher.networks.mlp import MLP
from serl_launcher.common.encoding import EncodingWrapper


def expectile_loss(diff, expectile=0.5):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def iql_value_loss(q, v, expectile):
    value_loss = expectile_loss(q - v, expectile)
    return value_loss.mean(), {
        "value_loss": value_loss.mean(),
        "uncentered_loss": jnp.mean((q - v) ** 2),
        "v": v.mean(),
    }


def iql_critic_loss(q, q_target):
    """mse loss"""
    critic_loss = jnp.square(q - q_target)
    return critic_loss.mean(), {
        "td_loss": critic_loss.mean(),
        "q": q.mean(),
    }


def awr_actor_loss(q, v, dist, actions, temperature=1.0, adv_clip_max=100.0, mask=None):
    adv = q - v

    exp_adv = jnp.exp(adv * temperature)
    exp_adv = jnp.minimum(exp_adv, adv_clip_max)

    log_probs = dist.log_prob(actions)
    actor_loss = -(exp_adv * log_probs)

    if mask is not None:
        actor_loss *= mask
        actor_loss = jnp.sum(actor_loss) / jnp.sum(mask)
    else:
        actor_loss = jnp.mean(actor_loss)

    behavior_mse = jnp.square(dist.mode() - actions).sum(-1)

    return actor_loss, {
        "actor_loss": actor_loss,
        "behavior_logprob": log_probs.mean(),
        "behavior_entropy": -log_probs.mean(),
        "behavior_mse": behavior_mse.mean(),
        "adv_mean": adv.mean(),
        "adv_max": adv.max(),
        "adv_min": adv.min(),
        "predicted actions": dist.mode(),
        "dataset actions": actions,
    }


def ddpg_bc_actor_loss(q, dist, actions, bc_loss_weight, mask=None):
    ddpg_objective = q  # policy action values
    bc_loss = -dist.log_prob(actions)
    actor_loss = -ddpg_objective + bc_loss_weight * bc_loss
    if mask is not None:
        actor_loss *= mask
        actor_loss = jnp.sum(actor_loss) / jnp.sum(mask)
    else:
        actor_loss = jnp.mean(actor_loss)
    return actor_loss, {
        "bc_loss": bc_loss.mean(),
        "ddpg_objective": ddpg_objective.mean(),
        "actor_loss": actor_loss,
    }


class IQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients)
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        qs = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

        return qs

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, train=False, grad_params=self.state.target_params
        )

    @partial(jax.jit, static_argnames="train")
    def forward_value(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for value network.
        Pass grad_params
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="value",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_value(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target value network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_value(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def _get_ensemble_q_value(self, q, rng):
        """
        subsample to a single critic value given an ensemble
        """
        if self.config["critic_subsample_size"] is not None:
            # REDQ
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            q = q[subsample_idcs]
            q = jnp.min(q, axis=0)
        else:
            # double Q
            q = jnp.min(q, axis=0)
        return q, rng

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]

        rng, key = jax.random.split(rng)
        next_v = self.forward_value(batch["next_observations"], key)
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        chex.assert_shape(target_q, (batch_size,))

        rng, key = jax.random.split(rng)
        q = self.forward_critic(
            batch["observations"],
            batch["actions"][..., :-1],
            key,
            grad_params=params,
        )
        chex.assert_shape(q, (self.config["critic_ensemble_size"], batch_size))

        # MSE loss
        critic_loss = jnp.square(q - target_q)
        chex.assert_shape(
            critic_loss, (self.config["critic_ensemble_size"], batch_size)
        )

        return critic_loss.mean(), {
            "td_loss": critic_loss.mean(),
            "q": q.mean(),
            "target_q": target_q.mean(),
        }

    def value_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, key = jax.random.split(rng)
        q = self.forward_target_critic(
            batch["observations"], batch["actions"][..., :-1], key
        )  # no gradient
        q, rng = self._get_ensemble_q_value(q, rng)  # min over Q functions

        rng, key = jax.random.split(rng)
        v = self.forward_value(batch["observations"], key, grad_params=params)

        # expectile loss
        return iql_value_loss(q, v, self.config["expectile"])

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, key = jax.random.split(rng)

        if self.config["update_actor_with_target_adv"]:
            critic_fn = self.forward_target_critic
        else:
            # Seohong: not using the target will make updates faster
            critic_fn = self.forward_critic

        rng, key = jax.random.split(rng)
        dist = self.forward_policy(batch["observations"], key, grad_params=params)
        mask = batch.get("actor_loss_mask", None)

        if self.config["actor_type"] == "awr":

            q = critic_fn(batch["observations"], batch["actions"][..., :-1], key)  # no gradient
            q, rng = self._get_ensemble_q_value(q, rng)  # min over Q functions

            rng, key = jax.random.split(rng)
            v = self.forward_value(batch["observations"], key)  # no gradients

            return awr_actor_loss(
                q,
                v,
                dist,
                batch["actions"][..., :-1],
                self.config["temperature"],
                mask=mask,
            )
        elif self.config["actor_type"] == "ddpg+bc":

            rng, key = jax.random.split(rng)
            policy_a = dist.sample(seed=key)
            policy_q = critic_fn(batch["observations"], policy_a)  # no gradient
            policy_q, rng = self._get_ensemble_q_value(
                policy_q, rng
            )  # min over Q functions

            return ddpg_bc_actor_loss(
                policy_q,
                dist,
                batch["actions"][..., :-1],
                self.config["actor_bc_loss_weight"],
                mask=mask,
            )
        else:
            raise NotImplementedError

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        rng, new_rng = jax.random.split(self.state.rng)
        batch_size = batch["rewards"].shape[0]

        loss_fns = {
            "critic": partial(self.critic_loss_fn, batch),
            "value": partial(self.value_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
        }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # update rng
        new_state = new_state.replace(rng=new_rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.forward_policy(observations, seed, train=False)
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):

        dist = self.forward_policy(batch["observations"], train=False)
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"][..., :-1])
        mse = ((pi_actions - batch["actions"][..., :-1]) ** 2).mean()

        v = self.forward_value(batch["observations"], train=False)
        next_v = self.forward_value(batch["next_observations"], train=False)
        target = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        q = self.forward_critic(batch["observations"], batch["actions"][..., :-1], train=False)
        q, _ = self._get_ensemble_q_value(q, self.state.rng)  # min over Q functions
        q_target = self.forward_target_critic(batch["observations"], batch["actions"][..., :-1])
        q_target, _ = self._get_ensemble_q_value(
            q_target, self.state.rng
        )  # min over Q functions

        metrics = {
            "log_probs": log_probs,
            "action_mse": mse,
            "pi_actions": pi_actions,
            "v": v.mean(),
            "q": q.mean(),
            "value loss": expectile_loss(q_target - v, self.config["expectile"]).mean(),
            "critic mse loss": jnp.square(target - q).mean(),
            "advantage": target - v,
            "qf_advantage": q - v,
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        network_type: str = "mlp",
        encoder_type: str = "resnet-pretrained",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        shared_encoder: bool = True,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "kernel_init_type": "var_scaling",
            "kernel_scale_final": 1e-2,
        },
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "kernel_init_type": "var_scaling",
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
        },
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        value_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount=0.99,
        expectile=0.9,
        temperature=1.0,
        target_update_rate=0.005,
        update_actor_with_target_adv=True,
        actor_type="awr",
        actor_bc_loss_weight=0.0,
        **kwargs,
    ):
        assert actor_type in ("awr", "ddpg+bc")
        assert not (
            actor_bc_loss_weight > 0 and actor_type == "awr"
        ), "BC loss is not yet supported with AWR"

        # update algorithm config
        config = ConfigDict(kwargs)
        config["image_keys"] = image_keys

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        networks = {
            "actor": Policy(
                encoder_def,
                MLP(**policy_network_kwargs),
                action_dim=actions[..., :-1].shape[-1],
                **policy_kwargs,
            ),
            "value": ValueCritic(encoder_def, MLP(**critic_network_kwargs)),
            "critic": Critic(
                encoder_def,
                network=ensemblize(
                    partial(MLP, **critic_network_kwargs), critic_ensemble_size
                )(name="critic_ensemble"),
            ),
        }

        model_def = ModuleDict(networks)

        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "value": make_optimizer(**value_critic_optimizer_kwargs),
            "critic": make_optimizer(**value_critic_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            value=[observations],
            critic=[observations, actions[..., :-1]],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                expectile=expectile,
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                update_actor_with_target_adv=update_actor_with_target_adv,
                actor_type=actor_type,
                actor_bc_loss_weight=actor_bc_loss_weight,
            )
        )
        return cls(state, config)
