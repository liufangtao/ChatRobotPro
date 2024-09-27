from collections import deque
import numpy as np


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


class PolicyState:
    def __init__(self, history_horizon=1):
        self._history_horizon = history_horizon
        self._state_ = {
            "history": {
                "horizon": history_horizon,
                "observations": deque(maxlen=history_horizon),
                "num_obs": 0,
            }
        }

    def Update(self, new_obs):
        """将observation累积到历史状态中"""
        history = self._state_["history"]
        history["observations"].append(new_obs)
        history["num_obs"] += 1

    def GetFullObservation(self):
        history = self._state_["history"]
        return stack_and_pad(history["observations"], history["num_obs"])


class BasePolicy:

    def CreateSession(self):
        raise NotImplemented
