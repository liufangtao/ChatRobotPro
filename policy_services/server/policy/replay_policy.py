import datetime
from functools import partial
import logging
import os
from typing import Any


import numpy as np
import dlimp as dl
import h5py
from omegaconf import DictConfig

from .base_policy import BasePolicy


class ReplayPolicy(BasePolicy):
    def __init__(
        self,
        episode_file_path,
        repeat=True,
        action: DictConfig = None,
        observation: DictConfig = None,
        discount: float = 1.0,
        **kwargs
    ) -> None:

        self._episode_file_path = episode_file_path
        self._action_cfg = action
        self._repeat = repeat
        self._observation = observation
        logging.info("Init ReplayPolicy")

    def _make_action_generator_(self):
        root = h5py.File(self._episode_file_path, "r")
        cfg = self._action_cfg
        loc = cfg.location
        horizon = cfg.horizon
        shift = max(cfg.shift, 1) if cfg.delta else cfg.shift
        # actions = root[self._action_cfg.location]
        nsteps = len(root[cfg.location]) - horizon - shift

        while True:
            for start_ts in range(nsteps):
                # yield actions[start_ts:start_ts+self._action_cfg.horizon]
                if cfg.delta:
                    delta_mask = cfg.get("delta_mask", None) or [1] * cfg.dim
                    delta_mask = np.array(delta_mask).astype(bool)
                    shifted_action = root[loc][
                        start_ts + shift : start_ts + shift + horizon
                    ]
                    current_action = root[loc][start_ts : start_ts + horizon]
                    actions = np.where(
                        delta_mask, shifted_action - current_action, shifted_action
                    )
                else:
                    shifted_action = root[loc][
                        start_ts + shift : start_ts + shift + horizon
                    ]
                    actions = shifted_action

                if cfg.get("noise", False) and cfg.noise.enabled:
                    noise = np.random.normal(
                        loc=cfg.noise.loc, scale=cfg.noise.scale, size=actions.shape
                    )
                    actions += noise

                yield actions

    def CreateSession(self):
        return dict(action_generator=self._make_action_generator_())

    def __call__(
        self,
        session: dict,
        observation: dict,
        language_instruction: str = None,
        unnorm_key: str = None,
        **kwargs
    ) -> Any:

        if session is None or not isinstance(session, dict):
            return None
        action_gen = session["action_generator"]
        action = next(action_gen)
        if self._action_cfg.delta:
            action *= self._action_cfg.discount

        action = self._action_smooth(action, session)
        return action

    def _action_smooth(self, actions, session: dict):
        self._action_smooth_window = 10
        self._action_output_chunk = 5
        chunck_size = actions.shape[-2]
        smooth_window = min(self._action_smooth_window, chunck_size)
        output_chunk = min(self._action_output_chunk, chunck_size - smooth_window)
        if "action_chuncks" not in session:
            self._smooth_weight = np.ones((smooth_window,)) / smooth_window
            session["action_chuncks"] = actions[np.newaxis, :, :].repeat(
                smooth_window, axis=0
            )

        A = session["action_chuncks"]
        A[: smooth_window - 1, : chunck_size - 1, :] = A[1:, 1:, :]
        A[-1, :, :] = actions
        session["action_chuncks"] = A
        smoothed_actions = self._smooth_weight.T.dot(
            A[:, :output_chunk, :].transpose(1, 0, 2)
        )

        return smoothed_actions
