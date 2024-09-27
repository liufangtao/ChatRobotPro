import json
import logging
import os
import random
import time
import gym
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def resize_image(rgb_array, new_size=None):

    if new_size == None or new_size[:2] == rgb_array.shape[:2]:
        return rgb_array
    image = Image.fromarray(rgb_array.astype(np.uint8), "RGB")
    image = image.resize(new_size[:2][::-1])
    return np.array(image)


class ReplayGymEnv(gym.Env):
    def __init__(self, config_file: str, **kwargs):
        # episode_file_path,
        # repeat=True,
        # observation:DictConfig=None,
        # im_size: int = 256,
        # seed: int = 5678,
        config = OmegaConf.load(config_file)

        self.observation_space = gym.spaces.Dict(
            {
                "images": gym.spaces.Dict(
                    {
                        img.name: gym.spaces.Box(
                            low=np.zeros(config.observations.images.image_size),
                            high=255 * np.ones(config.observations.images.image_size),
                            dtype=np.uint8,
                        )
                        for img in config.observations.images.list
                    }
                ),
                "proprioception": gym.spaces.Dict(
                    {
                        s.name: gym.spaces.Box(
                            low=np.ones((s.dim,)) * -1,
                            high=np.ones((s.dim,)),
                            dtype=np.float32,
                        )
                        for s in config.observations.states
                    }
                ),
                "time_step": gym.spaces.Discrete(2**30),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )

        self._episode_file_path = config.episode_file_path
        self._repeat = config.repeat
        self._observation_info = config.observations
        self._action_info = config.action
        self._config = config

        # self._im_size = im_size

        self._observation_generator = self._make_observation_generator_()

        self._previous_step_info = None
        from collections import deque

        self._history_errors = deque(maxlen=50)

        if "action_encoding" in kwargs:
            self._action_info.encoding = kwargs["action_encoding"]

    def _make_observation_generator_(self):
        root = h5py.File(self._episode_file_path, "r")
        state_location = {a.name: a.location for a in self._observation_info.states}
        image_location = {
            img.name: img.location for img in self._observation_info.images.list
        }
        # depth_location = {img.name: img.location for img in self._observation_info.images_depth.list}
        image_resize = self._observation_info.images.image_size
        # image_depth_resize = self._observation_info.images_depth.image_size

        episode_len = root[self._observation_info.images.list[0].location].shape[0]

        delay = 1 / (self._config.fps)
        time_step = 0
        i = 0
        start = time.time()
        while True:
            for _ in range(episode_len):
                stride = self._config.time_step.stride + random.randint(
                    *self._config.time_step.random
                )
                # i += stride
                time_step += stride
                i = time_step % episode_len

                
                images = {
                    k: resize_image(root[loc][i], image_resize)
                    for k, loc in image_location.items()
                }
                states = {k: root[loc][i] for k, loc in state_location.items()}

                # images_depth = {k: resize_image(root[loc][i], image_depth_resize) for k,loc in image_depth_location.items()}

                loc = self._action_info.location
                horizon = self._action_info.horizon
                shift = (
                    max(self._action_info.shift, 1)
                    if self._action_info.delta
                    else self._action_info.shift
                )
                if i + shift + horizon >= episode_len:
                    break
                if self._action_info.delta:
                    delta_mask = (
                        self._action_info.get("delta_mask", None)
                        or [1] * self._action_info.dim
                    )
                    delta_mask = np.array(delta_mask).astype(bool)
                    shifted_action = root[loc][i + shift : i + shift + horizon]
                    current_action = root[loc][i : i + horizon]
                    action = np.where(
                        delta_mask, shifted_action - current_action, shifted_action
                    )
                else:
                    shifted_action = root[loc][i + shift : i + shift + horizon]
                    action = shifted_action

                # yield actions
                actions = {self._action_info.name: action[0]}

                self._previous_step_info = {
                    "index": i,
                    **images,
                    # **images_depth,
                    **states,
                    **actions,
                    "time_step": time_step,
                }

                yield time_step, images, states, actions
                cost = time.time() - start
                #logging.debug(f"cost={cost:.3f}s")
                
                time.sleep(max(0, delay - cost))
                start = time.time()
                

    def analyze_action(self, action):
        def array_to_string(arr, precision=2):
            formater = f"{{:.{precision}f}}"  # .format(precision)
            # 使用numpy的array2string方法，设置精度和分隔符，避免换行
            arr_str = np.array2string(
                arr, precision=precision, separator=",", suppress_small=True
            )
            # 去掉字符串两端的方括号
            # arr_str = arr_str[1:-1]
            return arr_str.replace("\n", " ")

        mask = np.array(self._config.action.delta_mask).astype(dtype=bool)
        if self._previous_step_info:
            # if self._config.action.encoding == "JOINT_POS_BIMANUAL":
            #    delta = action
            # elif self._config.action.encoding == "JOINT_POS_BIMANUAL_ABS":
            #    delta = action - self._previous_step_info['action']
            time_step = self._previous_step_info.get("time_step", None)
            state = self._previous_step_info[self._observation_info.states[-1].name]
            expected_action = self._previous_step_info[self._action_info.name]
            error = action - expected_action
            # print(f"ReplayGymEnv: delta_action = {array_to_string(delta)}")
            self._history_errors.append(error)

            stacked_arrays = np.stack(self._history_errors, axis=0)  # [:,mask]
            mean_error = np.mean(stacked_arrays, axis=0)
            variance_value = np.var(stacked_arrays, axis=0)

            # print(f"ReplayGymEnv: delta_action = {array_to_string(delta)}")
            # import numpy.ma as ma
            # masked_data = ma.array(delta, mask=~np.array(self._config.action.delta_mask))  # 使用~反转掩码，因为ma.array中True表示忽略
            # squared_sum = (masked_data**2).sum().sqrt()
            # 使用掩码选择元素

            # selected_data = delta[mask]

            def cosine_similarity(vec_a, vec_b):
                """计算两个向量之间的夹角余弦值"""
                dot_product = np.dot(vec_a, vec_b)
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                return dot_product / (norm_a * norm_b + epsilon)

            epsilon = 0.000001
            # 计算L2范数
            mse = np.linalg.norm(error[mask], 2)
            l2_norm_action = np.linalg.norm(action[mask], 2)
            l2_norm_expected = np.linalg.norm(expected_action[mask], 2)
            # cos = action[mask].dot(expected_action[mask])/(l2_norm_action*l2_norm_expected+epsilon)
            # cos = cosine_similarity(action[:3],expected_action[:3])
            cos = cosine_similarity(action[mask], expected_action[mask])
            mean_value = np.linalg.norm(mean_error[mask], 2)
            variance_value = np.linalg.norm(variance_value, 2)
            info = {
                "norm(mean(error))": f"{mean_value:.6f}",
                "norm(error)": f"{mse:.6f}",
                "norm(action)": f"{l2_norm_action:.6f}",
                "norm(action)": f"{l2_norm_action:.6f}",
                "norm(expect)": f"{l2_norm_expected:.6f}",
                "rate": f"{mse/(l2_norm_action+l2_norm_expected+epsilon):.6f}",
                "cos": f"{cos:.6f}",
                "error_": f"{array_to_string(mean_error)}",
                "error ": f"{array_to_string(error)}",
                "action": f"{array_to_string(action)}",
                "expect": f"{array_to_string(expected_action)}",
                "propri": f"{array_to_string(state)}",
                "time_step": time_step,
            }
            # info = json.dumps(info,ensure_ascii=False,indent=1)
            info = "\n".join([f"{k}:{v}" for k, v in info.items()])
            logging.debug(f"ReplayGymEnv: {info}")
            pass

        # self._previous_step_info = dict(
        #    action = action,
        #    obs = obs,
        # )

    def step(self, action):
        start_analyze = time.time()
        self.analyze_action(action=action)
        cost_analyze = time.time() - start_analyze
        # print(f'action={action}')
        # obs = self.get_obs()
        start_observation = time.time()
        time_step, images, states, actions = next(self._observation_generator)
        cost_observe = time.time() - start_observation
        cost = dict(
            analyze = f"{cost_analyze:.3f}s",
            observe = f"{cost_observe:.3f}s",
        )
        logging.debug(f"evn.step: {cost}")
        
        obs = {
            "images": images,
            "proprioception": states,
            "time_step": time_step,
        }
        reward = 0
        info = {}

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        # obs = self.get_obs()
        self._observation_generator = self._make_observation_generator_()
        time_step, images, states, actions = next(self._observation_generator)
        obs = {
            "images": images,
            "proprioception": states,
            "time_step": time_step,
        }
        info = {}
        self._episode_is_success = 0
        # print(f"####obs={obs},info={info}")
        return obs, info


# register gym environments
gym.register(
    "aloha-replay-v0",
    entry_point=lambda: ReplayGymEnv(
        config_file=os.path.join(os.path.dirname(__file__), "config_replay.yaml")
    ),
)
