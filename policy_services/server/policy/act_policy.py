
import os
import math
import time
import torch
import pickle
import argparse
import collections
import numpy as np

from typing import Any
from einops import rearrange

import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', "robot_envs/real_envs/cobot_magic/aloha-devel/act"))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', "robot_envs/real_envs/cobot_magic/aloha-devel"))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy_act import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

from .base_policy import BasePolicy,PolicyState

class AllTimeAction:
    def __init__(self, max_publish_step, chunk_size, state_dim):
        self._max_publish_step = max_publish_step
        self._chunk_size = chunk_size
        self._state_dim = state_dim
        self.all_time_actions = []
        self.reset(self._max_publish_step, self._chunk_size, self._state_dim)
    
    def reset(self, max_publish_step, chunk_size, state_dim):
        self._max_publish_step = max_publish_step
        self._chunk_size = chunk_size
        self._state_dim = state_dim
        self.all_time_actions = np.zeros([
            max_publish_step,
            max_publish_step + chunk_size,
            state_dim
        ])
        self.t = 0
        self.max_t = 0

    def update_actions(self, actions):
        self.all_time_actions[[self.t], self.t:self.t + self._chunk_size] = actions

class ActPolicy(BasePolicy):
    def __init__(self,
                 model_path,
                 pkl_path,
                 history_horizon=1,
                 robot_type=2,
                 ):
        self._model_path = model_path
        self._pkl_path = pkl_path
        self._history_horizon = history_horizon
        self._robot_type = robot_type
        self.args = self.get_arguments()
        self.config = self.get_model_config(self.args)
        self._policy = None
        self._stats = None
        self._max_publish_step = self.config['episode_len']
        self._chunk_size = self.config['policy_config']['chunk_size']
        self._pre_joint_process = None
        self._pre_eef_process = None
        self._post_joint_process = None
        self._post_eef_process = None
        self._pre_action = None
        self.init_model()
    
    def init_model(self):
        set_seed(1000)

        # 1 创建模型数据  继承nn.Module
        self._policy = self.make_policy(self.config['policy_class'], self.config['policy_config'])

        # 2 加载模型权重
        ckpt_path = os.path.join(self._model_path)
        state_dict = torch.load(ckpt_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
                continue
            if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
                continue
            new_state_dict[key] = value
        loading_status = self._policy.deserialize(new_state_dict)
        if not loading_status:
            print("ckpt path not exist")
            return False

        # 3 模型设置为cuda模式和验证模式
        self._policy.cuda()
        self._policy.eval()

        # 4 加载统计值
        stats_path = os.path.join(self._pkl_path)
        # 统计的数据  # 加载action_mean, action_std, state_mean, state_std 14维
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            self._stats = stats

        # 数据预处理和后处理函数定义
        if self.args.use_joint:
            if "state_mean" in stats.keys():
                self._pre_joint_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
            else:
                self._pre_joint_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        else:
            if "state_mean" in stats.keys():
                self._pre_eef_process = lambda next_action: (next_action - stats["state_mean"]) / stats["state_std"]
            else:
                self._pre_eef_process = lambda next_action: (next_action - stats["ee_mean"]) / stats["ee_std"]
        if "state_mean" in stats.keys():
            self._pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
        else:
            self._pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

        if "state_mean" in stats.keys():
            self._post_joint_process = lambda a: a * stats['state_std'] + stats['state_mean']
        else:
            self._post_joint_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

        if "state_mean" in stats.keys():
            self._post_eef_process = lambda a: a * stats['state_std'] + stats['state_mean']
        else:
            self._post_eef_process = lambda a: a * stats['ee_std'] + stats['ee_mean']

    def make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        elif policy_class == 'Diffusion':
            policy = DiffusionPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy
    
    def CreateSession(self):        
        return AllTimeAction(self._max_publish_step, self._chunk_size, self.config['state_dim'])
    
    def get_observation(self, observation):
        obs = collections.OrderedDict()
        image_dict = dict()
        # front
        image_dict[self.config['camera_names'][0]] = observation['images']['front']
        # left
        image_dict[self.config['camera_names'][1]] = observation['images']['left_wrist']
        # right
        image_dict[self.config['camera_names'][2]] = observation['images']['right_wrist']
        # head
        if self._robot_type == 2:
            image_dict[self.config['camera_names'][3]] = observation['images']['head']

        obs['images'] = image_dict

        if self.args.use_depth_image:
            image_depth_dict = dict()
            # image_depth_dict[self.config['camera_names'][0]] = img_front_depth
            # image_depth_dict[self.config['camera_names'][1]] = img_left_depth
            # image_depth_dict[self.config['camera_names'][2]] = img_right_depth
            obs['images_depth'] = image_depth_dict

        if self.args.use_joint:
            obs['qpos'] = observation['states']['joints_state']
        else:
            obs['ee'] = observation['states']['joints_state']
        
        if self.args.use_robot_base:
            # obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            # if self.args.use_joint:
            #     obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
            # else:
            #     obs['ee'] = np.concatenate((obs['ee'], obs['base_vel']), axis=0)
            pass
        else:
            obs['base_vel'] = [0.0, 0.0]

        return obs
    
    def get_image(self, observation, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image
    
    def get_depth_image(self, observation, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_images.append(observation['images_depth'][cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image
    
    def actions_interpolation(self, args, pre_action, actions, stats):
        steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
        if "state_mean" in stats.keys():
            pre_process = lambda s_qpos: (s_qpos - stats['state_mean']) / stats['state_std']
        else:
            pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

        if "state_mean" in stats.keys():
            post_process = lambda a: a * stats['state_std'] + stats['state_mean']
        else:
            post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']

        result = [pre_action]
        post_action = post_process(actions[0])
        # print("pre_action:", pre_action[7:])
        # print("actions_interpolation1:", post_action[:, 7:])
        max_diff_index = 0
        max_diff = -1
        for i in range(post_action.shape[0]):
            diff = 0
            for j in range(pre_action.shape[0]):
                if j == 6 or j == 13:
                    continue
                diff += math.fabs(pre_action[j] - post_action[i][j])
            if diff > max_diff:
                max_diff = diff
                max_diff_index = i

        for i in range(max_diff_index, post_action.shape[0]):
            step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j])/steps[j]) for j in range(pre_action.shape[0])])
            inter = np.linspace(result[-1], post_action[i], step+2)
            result.extend(inter[1:])
        while len(result) < args.chunk_size+1:
            result.append(result[-1])
        result = np.array(result)[1:args.chunk_size+1]
        # print("actions_interpolation2:", result.shape, result[:, 7:])
        result = pre_process(result)
        result = result[np.newaxis, :]
        return result
    
    def inference_process(self, obs):
         # 归一化处理qpos 并转到cuda
        if self.args.use_joint:
            qpos = self._pre_joint_process(obs['qpos'])
        else:
            qpos = self._pre_eef_process(obs['ee'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # 当前图像curr_image获取图像
        curr_image = self.get_image(obs, self.config['camera_names'])
        curr_depth_image = None
        if self.args.use_depth_image:
            curr_depth_image = self.get_depth_image(obs, self.config['camera_names'])

        start_time = time.time()
        all_actions = self._policy(curr_image, curr_depth_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)

        inference_actions = all_actions.cpu().detach().numpy()
        if self._pre_action is None:
            if self.args.use_joint:
                self._pre_action = obs['qpos']
            else:
                self._pre_action = obs['ee']
        # print("obs['qpos']:", obs['qpos'][7:])
        if self.args.use_actions_interpolation:
            inference_actions = self.actions_interpolation(self.args, self._pre_action, inference_actions, self._stats)

        return inference_actions

    def __call__(self, session:PolicyState, observation:dict, language_instruction:str=None, unnorm_key:str=None, **kwargs) -> Any:
        with torch.inference_mode():
            assert isinstance(session, AllTimeAction)

            if self.config['temporal_agg']:
                if session.t >= self._max_publish_step:
                    session.reset(self._max_publish_step, self._chunk_size, self.config['state_dim'])
            
            obs = self.get_observation(observation)
                    
            if session.t >= session.max_t:        
                inference_actions = self.inference_process(obs)
                if inference_actions is not None:
                    all_actions = inference_actions
                    session.max_t = session.t + self.args.pos_lookahead_step
                    if self.config['temporal_agg']:
                        session.update_actions(all_actions)

            if self.config['temporal_agg']:
                actions_for_curr_step = session.all_time_actions[:, session.t]
                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights[:, np.newaxis]
                raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
            else:
                if self.args.pos_lookahead_step != 0:
                    raw_action = all_actions[:, session.t % self.args.pos_lookahead_step]
                else:
                    raw_action = all_actions[:, session.t % self._chunk_size]
            
            if self.args.use_joint:
                action = self._post_joint_process(raw_action[0])
            else:
                action = self._post_eef_process(raw_action[0])
            
            session.t += 1
        self._pre_action = action
        return action.reshape((1, 14))

    def get_model_config(self, args):
        if self._robot_type == 2:
            task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_head']}
        else:
            task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

        # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
        set_seed(1)
    
        # 如果是ACT策略
        # fixed parameters
        if args.policy_class == 'ACT':
            policy_config = {'lr': args.lr,
                            'lr_backbone': args.lr_backbone,
                            'backbone': args.backbone,
                            'masks': args.masks,
                            'weight_decay': args.weight_decay,
                            'dilation': args.dilation,
                            'position_embedding': args.position_embedding,
                            'loss_function': args.loss_function,
                            'chunk_size': args.chunk_size,     # 查询
                            'camera_names': task_config['camera_names'],
                            'use_depth_image': args.use_depth_image,
                            'use_robot_base': args.use_robot_base,
                            'kl_weight': args.kl_weight,        # kl散度权重
                            'hidden_dim': args.hidden_dim,      # 隐藏层维度
                            'dim_feedforward': args.dim_feedforward,
                            'enc_layers': args.enc_layers,
                            'dec_layers': args.dec_layers,
                            'nheads': args.nheads,
                            'dropout': args.dropout,
                            'pre_norm': args.pre_norm
                            }
        else:
            raise NotImplementedError

        config = {
            'ckpt_dir': args.ckpt_dir,
            'ckpt_name': args.ckpt_name,
            'ckpt_stats_name': args.ckpt_stats_name,
            'episode_len': args.max_publish_step,
            'state_dim': args.state_dim,
            'policy_class': args.policy_class,
            'policy_config': policy_config,
            'temporal_agg': args.temporal_agg,
            'camera_names': task_config['camera_names'],
        }
        return config

    def get_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
        parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
        parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
        parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
        parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
        parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
        parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
        parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
        parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
        parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
        parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features", required=False)
        parser.add_argument('--masks', action='store_true',
                            help="Train segmentation head if the flag is provided")
        parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
        parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
        parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
        parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

        parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
        parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
        parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
        parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
        parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
        parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
        parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
        parser.add_argument('--pre_norm', action='store_true', required=False)

        parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
        parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                            default=0, required=False)
        parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                            default=32, required=False)
        parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                            default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

        parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                            default=False, required=False)
        parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                            default=False, required=False)
        
        parser.add_argument('--use_joint', action='store', type=bool, help='use_joint',
                            default=True, required=False)  
        parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                            default=False, required=False)
        parser.add_argument('--use_head', action='store', type=bool, help='use_head',
                            default=True, required=False)
        
        parser.add_argument('--config', action='store', type=str, help='config',
                            default=True, required=False)
        
        args = parser.parse_args()
        return args
    
    

