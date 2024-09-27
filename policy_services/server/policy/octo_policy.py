from functools import partial
import logging
import random
import time
from typing import Any
from functools import lru_cache

import jax
import numpy as np
import dlimp as dl

from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng

from .base_policy import BasePolicy, PolicyState

from .utils import SingleWorkerExecutor


class OctoPolicy(BasePolicy):
    def __init__(
        self,
        model_path,
        language_encoder_model="/mnt/nas03/models/opensource/t5-base",
        pixel_keys: list = None,
        proprio_key: str = None,
        pixel_key_map: dict = None,
        proprio_key_map: dict = None,
        image_size=[256, 256],
        image_resize_size=None,
        history_horizon=1,
        discount=1.0,
        model_step: int = None,
        action_abs2delta=False,
        action_smooth_window: int = 32,
        action_output_chunk: int = 32,
        action_stride: int = 2,
        enable_pipeline_mode: bool = True,
        **kwargs,
    ) -> None:

        if pixel_key_map is None:
            # self._pixel_key_map = pixel_key_map
            pixel_key_map = {k: k for k in pixel_keys}
        if proprio_key_map is None:
            proprio_key_map = {proprio_key: proprio_key}
        self._pixel_key_map = pixel_key_map
        self._proprio_key_map = proprio_key_map
        # self._pixel_keys = pixel_keys
        # self._proprio_key = proprio_key
        self._model_path = model_path
        if isinstance(image_size, (float, int)):
            image_size = (image_size, image_size)
        # self._im_size = image_size
        if image_resize_size is None:
            image_resize_size = {k: image_size for k in pixel_key_map}

        self._image_resize_size = image_resize_size
        self._history_horizon = history_horizon
        self._discount = discount
        self._action_abs2delta = action_abs2delta
        self._action_smooth_window = action_smooth_window
        self._action_output_chunk = action_output_chunk
        self._action_stride = action_stride
        self._enable_pipeline_mode = enable_pipeline_mode
        # model_path = config["model_path"]
        # load finetuned model
        logging.info(f"Loading model {model_path}/{model_step or ''}")
        self._model = OctoModel.load_pretrained(model_path, model_step)
        # self._observation_keys = set(self._model.config["model"]["observation_tokenizers"].keys())
        self._warm_up()

        logging.info("OctoPolicy initialized!")

    def _warm_up(self):
        logging.info("OctoPolicy Warming Up...")
        obs = {
            "images": dict(),
            "states": dict(),
        }

        for _, image_name in self._pixel_key_map.items():
            random_images = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            obs["images"][image_name] = random_images

        for _, statee_name in self._proprio_key_map.items():
            random_state = np.random.rand(14)
            obs["states"][statee_name] = random_state

        session = self.CreateSession()
        from tqdm import trange


        obs["time_step"]=1
        self.__call__(session, observation=obs, language_instruction="test")
        
        for _ in trange(100, desc="测速"):
            obs["time_step"] += random.randint(1,5)
            self.__call__(session, observation=obs, language_instruction="test")

        logging.info("OctoPolicy Warming Up Done!")

    def CreateSession(self):
        return PolicyState(history_horizon=self._history_horizon)

    
    def __call__(
        self,
        session: PolicyState,
        observation: dict,
        language_instruction: str = None,
        unnorm_key: str = None,
        **kwargs,
    ) -> Any:

        assert isinstance(session, PolicyState)

        session.Update(
            {
                **{
                    new_key: observation["images"][old_key]
                    for new_key, old_key in self._pixel_key_map.items()
                },
                **{
                    new_key: observation["states"][old_key]
                    for new_key, old_key in self._proprio_key_map.items()
                },
            }
        )
        stacked_observation = session.GetFullObservation()

        predict_actions = partial(
            self._predict_actions,
            session=session,
            stacked_observation=stacked_observation,
            language_instruction=language_instruction,
            unnorm_key=unnorm_key,
            time_step=observation.get("time_step", None),
        )

        if self._enable_pipeline_mode:
            if "woker_excutor" not in session._state_:
                worker = SingleWorkerExecutor()
                worker.submit(predict_actions,time_step=-100)  # 第一次的时候故意重复提交一次
                session._state_["woker_excutor"] = worker

            worker = session._state_["woker_excutor"]
            actions,time_step = worker.result()  # 先取上一次的结果，再提交下一次的任务
            worker.submit(predict_actions)
        else:
            actions,time_step = predict_actions()

        # self._predict_actions(stacked_observation,language_instruction=language_instruction,unnorm_key=unnorm_key)

        if self._action_abs2delta:
            state = observation[self._proprio_key]
            actions = actions - state

        if self._enable_pipeline_mode:
            if time_step is None:
               stride = 1
            else:
                stride = observation.get("time_step", time_step+1)-time_step
                stride = min(stride, len(actions)-1)
            actions = actions[stride:]
        return actions
    

    def _predict_actions(
        self,
        session,
        stacked_observation,
        language_instruction: str = None,
        unnorm_key: str = None,
        time_step: int = None,
    ):
        
        start = time.time()
        dataset_statistics = self._get_dataset_statistics(unnorm_key)

        obs = self._observation_preprocess(stacked_observation, dataset_statistics)

        policy_fn = self._model.sample_actions
        # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
        policy_fn = supply_rng(policy_fn)
        # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
        batched_obs = jax.tree_map(lambda x: x[None], obs)
        self._use_cache = True
        if self._use_cache:
            task = self._create_task(language_instruction)
        else:
            task = self._model.create_tasks(texts=[language_instruction])
        # task = self._create_task(language_instruction)
        actions = policy_fn(
            batched_obs,
            task,
            unnormalization_statistics=dataset_statistics["action"],
        )
        actions = actions[0]
        actions = jax.device_get(actions)
        actions = self._action_post_process(actions, dataset_statistics["action"])
        actions = self._action_smooth(actions, session, time_step=time_step)
        cost = time.time()-start
        logging.debug(f"Octo Prediction Cost={cost:.3f}s")
        return actions, time_step


    @lru_cache(maxsize=10)
    def _create_task(self, language_instruction: str):

        encoder = self._model.module.octo_transformer.task_tokenizers["language"]
        params = self._model.params["octo_transformer"]["task_tokenizers_language"]
        task = self._model.create_tasks(texts=[language_instruction])
        task["language_instruction"] = encoder.apply(
            {"params": params},
            observations=None,
            tasks=task,
            # method="hf_model",
        ).tokens  # .last_hidden_state
        return task

    def _normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data - metadata["mean"]) / (metadata["std"] + 1e-8),
            data,
        )

    def _observation_preprocess(self, obs, dataset_statistics):

        curr_obs = obs
        for k, v in self._image_resize_size.items():
            resized = dl.transforms.resize_images(curr_obs, match=k, size=v)
            curr_obs.update(resized)
        for k in self._pixel_key_map:
            curr_obs[k] = np.array(curr_obs[k])

        for k in self._proprio_key_map:
            if k in dataset_statistics:
                curr_obs[k] = self._normalize(curr_obs[k], dataset_statistics[k])

        return curr_obs

    def _get_dataset_statistics(self, unnorm_key: str = None):
        assert (unnorm_key == None) or (unnorm_key in self._model.dataset_statistics)

        if unnorm_key == None:
            pass

        if unnorm_key in self._model.dataset_statistics:
            dataset_statistics = self._model.dataset_statistics[unnorm_key]
            # dataset_statistics=model.dataset_statistics['sim_2env_transfer_cube_300']
        elif "action" not in self._model.dataset_statistics:
            # 取第一个
            dataset_statistics = next(iter(self._model.dataset_statistics.values()))
        else:
            dataset_statistics = self._model.dataset_statistics

        return dataset_statistics

    def _action_smooth(self, actions, session: PolicyState, time_step: int = None):

        chunck_size = actions.shape[-2]
        smooth_window = min(self._action_smooth_window, chunck_size)
        output_chunk = min(self._action_output_chunk, chunck_size - smooth_window + 1)
        if "action_chuncks" not in session._state_:
            self._smooth_weight = np.ones((smooth_window,)) / smooth_window
            A = actions[np.newaxis, :, :].repeat(smooth_window, axis=0)
            session._state_["action_chuncks"] = A
            session._state_["time_step"] = time_step or 1  # - 1
        else:

            A = session._state_["action_chuncks"]
            last_time_step = session._state_["time_step"]
            new_time_step = time_step or (last_time_step + self._action_stride)
            assert new_time_step > last_time_step, f"不满足 new_time_step ({new_time_step}) > last_time_step ({last_time_step})"
            stride = min(smooth_window, new_time_step - last_time_step)
            A[: smooth_window - stride, : chunck_size - stride, :] = A[
                stride:, stride:, :
            ]
            A[-stride:, :, :] = actions[np.newaxis, :, :].repeat(
                stride, axis=0
            )  # np.repeat(actions[np.newaxis,:],stride,axis=0)
            # A[-1,:,:]=actions

            session._state_["action_chuncks"] = A
            session._state_["time_step"] = new_time_step

        smoothed_actions = self._smooth_weight.T.dot(
            A[:, :output_chunk, :].transpose(1, 0, 2)
        )

        return smoothed_actions

    def _action_post_process(self, actions, unnormalization_statistics):
        return np.clip(
            actions,
            unnormalization_statistics["p01"],
            unnormalization_statistics["p99"],
        )
