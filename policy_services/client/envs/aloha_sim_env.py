import copy
from typing import List

import os
os.environ['MUJOCO_GL'] = 'egl' 

import dlimp as dl
import gym
import numpy as np

# need to put https://github.com/tonyzhaozh/act in your PATH for this import to work
from sim_env import BOX_POSE, make_sim_env

class AlohaGymEnv(gym.Env):
    def __init__(
        self,
        task,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 5678,
        **kwargs
    ):
        self._task = task
        self._env = make_sim_env(task)
        self._obs_img_names = ["primary", "left_wrist","right_wrist","angle"]
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in self._obs_img_names[: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)
        if "action_encoding" in kwargs:
            self._action_encoding = kwargs["action_encoding"]
        else:
            self._action_encoding = "JOINT_POS_BIMANUAL_ABS"
    
        self._last_obs = {}

    def step(self, action):
        if self._action_encoding=="JOINT_POS_BIMANUAL":
            last_state=self._last_obs["proprio"]
            action += last_state 
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        #print(f"####obs={obs},info={info}")
        self._last_obs = obs
        return obs, reward, False, False, info
    
    

    def reset(self, **kwargs):

        cube_pose = self.sample_box_pose()
        peg_pose,socket_pose = self.sample_insertion_pose()
        
        if self._task == "sim_transfer_cube":
            BOX_POSE[0] = cube_pose
        
        elif self._task == "sim_insertion":
            BOX_POSE[0] = np.concatenate([peg_pose,socket_pose])
            
        elif self._task == "sim_mix_transfer_and_insertion":
            BOX_POSE[0] = np.concatenate([peg_pose,socket_pose,cube_pose])
        else:
            pass

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0
        #print(f"####obs={obs},info={info}")
        self._last_obs = obs
        return obs, info
    
    
    def sample_box_pose(self):
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])
        cube_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])

        cube_quat = np.array([1, 0, 0, 0])
        return np.concatenate([cube_position, cube_quat])

    def sample_insertion_pose(self):
        # Peg
        x_range = [0.1, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])
        peg_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])

        peg_quat = np.array([1, 0, 0, 0])
        peg_pose = np.concatenate([peg_position, peg_quat])

        # Socket
        x_range = [-0.2, -0.1]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]

        ranges = np.vstack([x_range, y_range, z_range])
        socket_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])

        socket_quat = np.array([1, 0, 0, 0])
        socket_pose = np.concatenate([socket_position, socket_quat])

        return peg_pose, socket_pose

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        obs_img_names = self._obs_img_names
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = np.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = np.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        if self._task == "sim_transfer_cube":
            return {
                "language_instruction": ["pick up the green cube and hand it over"],
            }
        
        elif self._task == "sim_insertion":
            return {
                "language_instruction": ["insert the red peg into the blue socket"],
            }
        elif self._task == "sim_mix_transfer_and_insertion":
            return {
                "language_instruction": ["pick up the green cube and hand it over"],
            }
        
    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


# register gym environments
gym.register(
    "aloha-sim-cube-v0",
    entry_point=lambda: AlohaGymEnv(
        task="sim_transfer_cube",
        camera_names=["top","left_wrist","right_wrist"]
    ),
)

gym.register(
    "aloha-sim-insertion-v0",
    entry_point=lambda: AlohaGymEnv(
        task="sim_insertion",
        camera_names=["top","left_wrist","right_wrist"]
    ),
)

gym.register(
    "aloha-sim-mix-v0",
    entry_point=lambda: AlohaGymEnv(
        task="sim_mix_transfer_and_insertion",
        camera_names=["top","left_wrist","right_wrist","angle"]
    ),
)
