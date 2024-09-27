#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名：convert_ainno2cobot.py
创建时间：2024年9月20日
作者:liufangtao
地点：北京
描述：这个脚本是宪法数据转换hdf5格式数据。

模块内容：
    - main(): 主函数，程序的入口点。
"""

import re
import cv2
import numpy as np
import h5py
from PIL import Image
import os
import json
import h5py
from cv2 import VideoCapture
from cv2 import imwrite
# from schema.episode_dataclass import Episode, Metadata, Observation, Step
from ainnorobot_data.schema.episode_dataclass import Episode, Metadata, Observation, Step
# TODO schema add cam and revise cam No
import time
from multiprocessing import Pool
import imageio
import argparse

def mkdir_ifmissing(fld):
    if not os.path.isdir(fld):
        os.makedirs(fld)

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_video2img(video_path):
    cap = VideoCapture(video_path)
    if not cap.isOpened():
        print('无法打开视频文件')
        exit()

    image_array = []

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        image_array.append(frame)
    image_array = np.array(image_array)
    cap.release()
    return image_array

def save_metadata_part(root, data_json):
    string_dt = h5py.special_dtype(vlen=bytes)
    metadata = root.create_group('metadata')
    task_name = data_json["metadata"]["task_name"]
    sample_rate = data_json["metadata"]["sample_rate"]

    metadata.create_dataset('task_name', data=np.array([task_name], dtype='S'), dtype=string_dt)
    metadata.create_dataset('sample_rate', data=np.array([sample_rate], dtype='S'), dtype=string_dt)

def save_data(episode_datadict, dataset_path, use_depthimg_flag=0,min_depth_value=0, max_depth_value=3000):
    camera_rgb_names = ['camera_3_rgb_front', 'camera_1_rgb_left', 'camera_2_rgb_right', 'camera_5_rgb_head']
    camera_depth_names = ['camera_3_depth_front', 'camera_1_depth_left', 'camera_2_depth_right', 'camera_5_depth_head']
    data_size = len(episode_datadict['observations']['arm_joints_state'])
    data_dict = episode_datadict

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_rgb_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), compression='gzip',
                                    data=data_dict['observations']['images'][cam_name])
        if use_depthimg_flag:
            image_depth = obs.create_group('images_depth')
            
            for depth_cam_name in camera_depth_names:
                depth_data = data_dict['observations']['images_depth'][depth_cam_name]
                # 将深度图像的数值从0到255恢复到0到3000
                depth_data = depth_data.astype(np.float32) * (max_depth_value - min_depth_value) / 255.0 + min_depth_value
                depth_data[depth_data <= 20] = 0
                depth_data = depth_data.astype(np.uint16)
                
                depth_data = depth_data[:, :, :, 0]  # 去掉多余的维度
                # Ensure depth data is in the correct shape (data_size, 480, 640, 1)
                
                _ = image_depth.create_dataset(depth_cam_name, (data_size, 480, 640), dtype='uint16', chunks=(1, 480, 640), compression='gzip',
                                               data=depth_data)
                # print(data_dict['observations']['images_depth'][depth_cam_name].shape)
                # print(data_dict['observations']['images_depth'][depth_cam_name].dtype)

        
        _ = obs.create_dataset('arm_joints_state', (data_size, 14), data=np.array(data_dict['observations']['arm_joints_state']))
        _ = obs.create_dataset('arm_eef_state', (data_size, 14), data=np.array(data_dict['observations']['arm_eef_state']))
        _ = root.create_dataset('master_arm_joints_state', (data_size, 14), data=np.array(data_dict['observations']['master_arm_joints_state']))
        _ = root.create_dataset('master_arm_eef_state', (data_size, 14), data=np.array(data_dict['observations']['master_arm_eef_state']))
        _ = root.create_dataset('base_state', (data_size, 2), data=np.array(data_dict['observations']['base_state']))
    
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)

def create_datadict(episode_datainfo, ainno_data_folder,use_depthimg_flag=0):
    json_path = os.path.join(ainno_data_folder, episode_datainfo['jsonfile'][0])
    camera_paths = {
    'camera_5_rgb_head': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera5' in x, episode_datainfo['cam_rgb_files']))),
    'camera_3_rgb_front': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera3' in x, episode_datainfo['cam_rgb_files']))),
    'camera_1_rgb_left': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera1' in x, episode_datainfo['cam_rgb_files']))),
    'camera_2_rgb_right': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera2' in x, episode_datainfo['cam_rgb_files'])))
}

    if use_depthimg_flag:
        camera_paths.update({
            'camera_5_depth_head': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera5' in x, episode_datainfo['cam_depth_files']))),
            'camera_3_depth_front': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera3' in x, episode_datainfo['cam_depth_files']))),
            'camera_1_depth_left': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera1' in x, episode_datainfo['cam_depth_files']))),
            'camera_2_depth_right': os.path.join(ainno_data_folder, next(filter(lambda x: 'camera2' in x, episode_datainfo['cam_depth_files'])))
        })

    data_json = load_json(json_path)
    task_name = data_json["metadata"]["task_name"]
    episode_id = data_json["metadata"]["episode_id"]

    data_dict = {
        'observations': {
            'images': {},
            'images_depth': {},
            'arm_joints_state': [],
            "arm_eef_state": []
        },
        'base_state': [],
        'master_arm_joints_state': [],
        'master_arm_eef_state': []
    }

    step_datas = data_json['steps']
    observation_joints_list = []
    observation_eef_list = []
    observation_master_arm_joints_list = []
    observation_master_arm_eef_list = []
    base_state_list = []

    with Pool() as pool:
        results = pool.map(save_video2img, camera_paths.values())

    for cam_name, result in zip(camera_paths.keys(), results):
        if 'rgb' in cam_name:
            data_dict['observations']['images'][cam_name] = result
        else:
            data_dict['observations']['images_depth'][cam_name] = result

    for idx, step in enumerate(step_datas):
        arm1_joints_state = step['observation']['arm1_joints_state']
        arm2_joints_state = step['observation']['arm2_joints_state']
        arm1_eef_state = step['observation']['arm1_eef_state']
        arm2_eef_state = step['observation']['arm2_eef_state']
        arm1_gripper_state = [i * 5 for i in step['observation']["arm1_gripper_state"]]
        arm2_gripper_state = [i * 5 for i in step['observation']["arm2_gripper_state"]]
        joints_state = arm1_joints_state + arm1_gripper_state + arm2_joints_state + arm2_gripper_state
        eef_state = arm1_eef_state + arm1_gripper_state + arm2_eef_state + arm2_gripper_state

        observation_joints_list.append(joints_state)
        observation_eef_list.append(eef_state)

        master_arm1_joints_state = step['observation']['master_arm1_joints_state']
        master_arm2_joints_state = step['observation']['master_arm2_joints_state']
        master_arm1_eef_state = step['observation']['master_arm1_eef_state']
        master_arm2_eef_state = step['observation']['master_arm2_eef_state']
        master_arm1_gripper_state = [i * 5 for i in step['observation']["master_arm1_gripper_state"]]
        master_arm2_gripper_state = [i * 5 for i in step['observation']["master_arm2_gripper_state"]]
        master_arm_joints_state = master_arm1_joints_state  + master_arm1_gripper_state + master_arm2_joints_state + master_arm2_gripper_state
        master_arm_eef_state = master_arm1_eef_state + master_arm1_gripper_state + master_arm2_eef_state + master_arm2_gripper_state

        observation_master_arm_joints_list.append(master_arm_joints_state)
        observation_master_arm_eef_list.append(master_arm_eef_state)

        base_state = step['observation']["base_state"]
        base_state_list.append(base_state)

    data_dict['observations']['base_state'] = base_state_list
    data_dict['observations']['arm_joints_state'] = observation_joints_list
    data_dict['observations']['arm_eef_state'] = observation_eef_list
    data_dict['observations']['master_arm_joints_state'] = observation_master_arm_joints_list
    data_dict['observations']['master_arm_eef_state'] = observation_master_arm_eef_list
    return data_dict, episode_id


def main():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--ainno_data_folder', type=str, default="/home/lft/codes/datas/data20240926/testdatas",required=False,
                        help='Path to the ainno data folder')
    parser.add_argument('--output_data_folder', type=str, default="/home/lft/codes/datas/data20240926/testdatas-train",required=False,
                        help='Path to the ainno data folder')
    parser.add_argument('--use_depthimg_flag', type=int, default=1, required=False,
                        help='Path to the ainno data folder')

    args = parser.parse_args()

    ainno_data_folder = args.ainno_data_folder
    output_data_folder = args.output_data_folder
    use_depthimg_flag = args.use_depthimg_flag
    print(f"The ainno data folder is: {ainno_data_folder}")
    print(f"The output data folder is: {output_data_folder}")


    # ainno_data_folder = "/1T/datas/data20240821_1/transfer_rubber_balls-ainno"
    ouput_cobot_folder = output_data_folder + '-back4train'
    mkdir_ifmissing(ouput_cobot_folder)
    

    json2data_dict = {}
    for file in os.listdir(ainno_data_folder):
        filepath = os.path.join(ainno_data_folder, file)
        if os.path.isfile(filepath):
            if file.endswith('.json'):
                filehead = file.split('.json')[0]
                if filehead not in json2data_dict.keys():
                    json2data_dict[filehead] = {'jsonfile': [],
                                                'cam_rgb_files': [],
                                                'cam_depth_files': []}
                json2data_dict[filehead]['jsonfile'].append(file)
            elif file.endswith('.mp4'):
                filehead = file.split('_camera')[0]
                if filehead not in json2data_dict.keys():
                    json2data_dict[filehead] = {'jsonfile': [],
                                                'cam_rgb_files': [],
                                                'cam_depth_files': []}
                cam_type = file.split('.')[0][-3:]
                if cam_type == 'rgb':
                    json2data_dict[filehead]['cam_rgb_files'].append(file)
                else:
                    json2data_dict[filehead]['cam_depth_files'].append(file)

    for filehead in json2data_dict.keys():
        episode_datainfo = json2data_dict[filehead]
        if len(episode_datainfo['jsonfile']) == 1 and len(episode_datainfo['cam_rgb_files']) == 4 or len(episode_datainfo['cam_depth_files']) == 4:
            episode_datadict, _ = create_datadict(episode_datainfo, ainno_data_folder,use_depthimg_flag)
            output_path = os.path.join(ouput_cobot_folder, filehead)
            print(output_path)
            save_data(episode_datadict, output_path,use_depthimg_flag)

if __name__ == '__main__':
    main()