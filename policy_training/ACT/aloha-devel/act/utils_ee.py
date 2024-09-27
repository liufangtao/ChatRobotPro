import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import random
import IPython

e = IPython.embed
import cv2

class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, arm_delay_time,
                 use_depth_image, use_robot_base):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # 1000
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_depth_image = use_depth_image
        self.arm_delay_time = arm_delay_time
        self.use_robot_base = use_robot_base
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        # 读取数据
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            original_action_shape = root['/action'].shape
            max_action_len = original_action_shape[0]  # max_episode
            if self.use_robot_base:
                original_action_shape = (original_action_shape[0], original_action_shape[1] + 2)

            start_ts = np.random.choice(max_action_len)  # 随机抽取一个索引
            actions = root['/observations/ee'][1:]
            actions = np.append(actions, actions[-1][np.newaxis, :], axis=0)
            ee = root['/observations/ee'][start_ts]
            if self.use_robot_base:
                ee = np.concatenate((ee, root['/base_action'][start_ts]), axis=0)
            image_dict = dict()
            image_depth_dict = dict()
            for cam_name in self.camera_names:
                if is_compress:
                    decoded_image = root[f'/observations/images/{cam_name}'][start_ts]
                    image_dict[cam_name] = cv2.imdecode(decoded_image, 1)
                    # print(image_dict[cam_name].shape)
                    # exit(-1)
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

                if self.use_depth_image:
                    image_depth_dict[cam_name] = root[f'/observations/images_depth/{cam_name}'][start_ts]

            start_action = min(start_ts, max_action_len - 1)
            index = max(0, start_action - self.arm_delay_time)
            action = actions[index:]  # hack, to make timesteps more aligned
            if self.use_robot_base:
                action = np.concatenate((action, root['/base_action'][index:]), axis=1)
            action_len = max_action_len - index  # hack, to make timesteps more aligned

        self.is_sim = is_sim

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        action_is_pad = np.zeros(max_action_len)
        action_is_pad[action_len:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        image_depth_data = np.zeros(1, dtype=np.float32)
        if self.use_depth_image:
            all_cam_images_depth = []
            for cam_name in self.camera_names:
                all_cam_images_depth.append(image_depth_dict[cam_name])
            all_cam_images_depth = np.stack(all_cam_images_depth, axis=0)
            # construct observations
            image_depth_data = torch.from_numpy(all_cam_images_depth)
            # image_depth_data = torch.einsum('k h w c -> k c h w', image_depth_data)
            image_depth_data = image_depth_data / 255.0

        ee_data = torch.from_numpy(ee).float()
        ee_data = (ee_data - self.norm_stats["ee_mean"]) / self.norm_stats["ee_std"]
        action_data = torch.from_numpy(padded_action).float()
        action_is_pad = torch.from_numpy(action_is_pad).bool()
        action_data = (action_data - self.norm_stats["ee_mean"]) / self.norm_stats["ee_std"]

        # torch.set_printoptions(precision=10, sci_mode=False)
        # torch.set_printoptions(threshold=float('inf'))
        # print("ee_data:", ee_data[7:])
        # print("action_data:", action_data[:, 7:])
        # print(image_data.shape)
        return image_data, image_depth_data, ee_data, action_data, action_is_pad


def get_norm_stats(dataset_dir, num_episodes, use_robot_base):
    """
    从给定的数据集目录中读取多个episode的ee和action数据，并计算其标准化统计信息。
    """
    all_ee_data = []
    all_action_data = []

    # 遍历每个episode
    for episode_idx in range(num_episodes):
        # 拼接episode的路径
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        # 打开hdf5文件
        with h5py.File(dataset_path, 'r') as root:
            # 从hdf5文件中读取ee数据
            # qpos = root['/observations/qpos'][()]
            ee = root['/observations/ee'][()]
            # 从hdf5文件中读取qvel数据（但在这段代码中并未使用）
            qvel = root['/observations/qvel'][()]
            # 从hdf5文件中读取action数据
            action = root['/action'][()]

            # 如果使用机器人基座数据
            if use_robot_base:
                # 拼接ee和基座动作数据
                ee = np.concatenate((ee, root['/base_action'][()]), axis=1)
                # 拼接action和基座动作数据
                action = np.concatenate((action, root['/base_action'][()]), axis=1)

        # 将ee数据转换为torch张量并添加到列表中
        all_ee_data.append(torch.from_numpy(ee))
        # 将action数据转换为torch张量并添加到列表中
        all_action_data.append(torch.from_numpy(action))

    # 将ee数据列表转换为张量堆叠
    all_ee_data = torch.stack(all_ee_data)
    # 将action数据列表转换为张量堆叠
    all_action_data = torch.stack(all_action_data)
    # 这里的代码没有实际作用，可以忽略
    all_action_data = all_action_data

    # 标准化action数据
    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    # 截断action数据的标准差，避免除以0的情况
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # 标准化ee数据
    # normalize ee data
    ee_mean = all_ee_data.mean(dim=[0, 1], keepdim=True)
    ee_std = all_ee_data.std(dim=[0, 1], keepdim=True)
    # 截断ee数据的标准差，避免除以0的情况
    ee_std = torch.clip(ee_std, 1e-2, np.inf)  # clipping

    # 将统计信息存储到字典中
    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "ee_mean": ee_mean.numpy().squeeze(), "ee_std": ee_std.numpy().squeeze(),
             "example_ee": ee}

    return stats


def load_data(dataset_dir, num_episodes, arm_delay_time, use_depth_image,  # /1T/cobot_magic/aloha-devel/datas/ ,70, 0, False, 
              use_robot_base, camera_names, batch_size_train, batch_size_val):  # False, ['cam_high', 'cam_left_wrist', 'cam_right_wrist'], 8, 8
    print(f'\nData from: {dataset_dir}\n')

    # obtain train test split
    train_ratio = 0.8  # 数据集比例
    shuffled_indices = np.random.permutation(num_episodes)  # 打乱

    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for ee and action  返回均值和方差
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_robot_base)

    # construct dataset and dataloader 归一化处理  结构化处理数据
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                    use_depth_image, use_robot_base)

    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, arm_delay_time,
                                  use_depth_image, use_robot_base)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])

    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


# helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
