"""
Simple ReplayBufferDataStore implementation for RL module.

This is a minimal version of replay buffer that inherits from DataStoreBase
and returns data in the same format as used in train_rlpd.py.
"""

import numpy as np
from threading import Lock
from typing import Optional, Dict, Any
import gymnasium as gym
from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(DataStoreBase):
    """
    Simple replay buffer that inherits from DataStoreBase.

    This implementation stores transitions with the following keys:
    - observations: environment observations
    - actions: actions taken
    - next_observations: next environment observations
    - rewards: reward signals
    - masks: done masks (1.0 - done)
    - dones: episode termination flags
    - grasp_penalty: optional grasp penalty (if included)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        include_grasp_penalty: bool = False,
    ):
        super().__init__(capacity)

        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity
        self.include_grasp_penalty = include_grasp_penalty

        # Initialize storage arrays
        self.observations = self._init_observation_storage(observation_space, capacity)
        self.next_observations = self._init_observation_storage(
            observation_space, capacity
        )
        self.actions = np.empty(
            (capacity, *action_space.shape), dtype=action_space.dtype
        )
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=bool)

        if include_grasp_penalty:
            self.grasp_penalties = np.empty((capacity,), dtype=np.float32)

        # Buffer state
        self._size = 0
        self._insert_index = 0
        self._lock = Lock()

    def _init_observation_storage(self, obs_space: gym.Space, capacity: int):
        """Initialize storage for observations based on observation space."""
        if isinstance(obs_space, gym.spaces.Box):
            return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
        elif isinstance(obs_space, gym.spaces.Dict):
            storage = {}
            for key, space in obs_space.spaces.items():
                storage[key] = self._init_observation_storage(space, capacity)
            return storage
        else:
            raise NotImplementedError(
                f"Unsupported observation space type: {type(obs_space)}"
            )

    def _insert_observation(self, storage, data, index):
        """Recursively insert observation data into storage."""
        if isinstance(storage, np.ndarray):
            storage[index] = data
        elif isinstance(storage, dict):
            for key in storage.keys():
                self._insert_observation(storage[key], data[key], index)
        else:
            raise TypeError(f"Unsupported storage type: {type(storage)}")

    def _sample_observation(self, storage, indices):
        """Recursively sample observation data from storage."""
        if isinstance(storage, np.ndarray):
            return storage[indices]
        elif isinstance(storage, dict):
            sampled = {}
            for key in storage.keys():
                sampled[key] = self._sample_observation(storage[key], indices)
            return sampled
        else:
            raise TypeError(f"Unsupported storage type: {type(storage)}")

    def insert(self, transition: Dict[str, Any]):
        """Insert a transition into the replay buffer."""
        with self._lock:
            # Insert observations
            self._insert_observation(
                self.observations, transition["observations"], self._insert_index
            )
            self._insert_observation(
                self.next_observations,
                transition["next_observations"],
                self._insert_index,
            )

            # Insert other data
            self.actions[self._insert_index] = transition["actions"]
            self.rewards[self._insert_index] = transition["rewards"]
            self.masks[self._insert_index] = transition["masks"]
            self.dones[self._insert_index] = transition["dones"]

            if self.include_grasp_penalty and "grasp_penalty" in transition:
                self.grasp_penalties[self._insert_index] = transition["grasp_penalty"]

            # Update indices
            self._insert_index = (self._insert_index + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions from the replay buffer."""
        with self._lock:
            if self._size == 0:
                raise ValueError("Cannot sample from empty buffer")

            # Sample random indices
            indices = np.random.randint(0, self._size, size=batch_size)

            # Sample data
            batch = {
                "observations": self._sample_observation(self.observations, indices),
                "next_observations": self._sample_observation(
                    self.next_observations, indices
                ),
                "actions": self.actions[indices],
                "rewards": self.rewards[indices],
                "masks": self.masks[indices],
                "dones": self.dones[indices],
            }

            if self.include_grasp_penalty:
                batch["grasp_penalty"] = self.grasp_penalties[indices]

            return batch

    def get_iterator(self, sample_args: Dict[str, Any] = None, device=None):
        """Get an iterator for sampling batches."""
        if sample_args is None:
            sample_args = {"batch_size": 32}

        batch_size = sample_args.get("batch_size", 32)

        while True:
            batch = self.sample(batch_size)

            yield batch

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self._size

    def latest_data_id(self) -> int:
        """Return the latest data id (required by DataStoreBase)."""
        return self._insert_index

    def get_latest_data(self, from_id: int):
        """Get latest data from a specific id (required by DataStoreBase)."""
        # This is a simplified implementation
        # In practice, you might need to implement this based on specific requirements
        raise NotImplementedError("get_latest_data not implemented in simple version")
