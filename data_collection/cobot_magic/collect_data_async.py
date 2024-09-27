# -- coding: UTF-8
import os
import time
import json
import numpy as np
import h5py
import argparse
import dm_env

import psutil

import collections
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2

import threading
from threading import Event, Thread
import multiprocessing
from multiprocessing import Queue
import copy
import shutil

key_pressed = None 
on_process = False

cur_datetime = str()
user = str()
rate = str()
num_steps = 0
json_data = None
cur_task = []

def capture_key_thread(stop_event):  
    global key_pressed, on_process  # 声明为全局变量  
    try:  
        while not stop_event.is_set(): 
            if on_process:
                ch = input()
                if ch:  
                    key_pressed = ch  # 修改全局变量  
                    on_process = False
            else:
                time.sleep(1)
        
    finally: 
        pass

def get_date_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime())

def input_metadata_info(args):
    global user, rate, num_steps, json_data, cur_task
    user = input("请输入用户名: ")
    print("当前用户为：%s\n\n" % user)

    tasks = {}
    for entry in json_data["task_info"]:
        if entry["use"]:
            tasks[entry["task_id"]] = entry["display_name"]
    
    while True:
        input_task = "请选择任务名称：\n"
        for key in tasks:
            input_task += "[%s]%s\n" % (key, tasks[key])

        task_id = int(input(input_task))
        if task_id in tasks.keys():
            task_name = tasks[task_id]
            for entry in json_data["task_info"]:
                if entry["use"] and entry["display_name"] == task_name:
                    cur_task = entry
                    break
            print('选择的任务名称为：%s\n\n' % task_name)
            break
        else:
            print('任务选择错误')
            continue

    rate = args.frame_rate
    args.max_timesteps = cur_task["step_max"]

def save_metadata(root, cur_datetime, user, rate, num_steps, json_data, cur_task):
    string_dt = h5py.special_dtype(vlen=bytes)
    metadata = root.create_group('metadata')
    metadata.create_dataset('dataset_name', data=np.array([json_data['dataset_name'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('experiment_time', data=np.array([cur_datetime], dtype='S'), dtype=string_dt)
    metadata.create_dataset('operator', data=np.array([user], dtype='S'), dtype=string_dt)
    metadata.create_dataset('scene', data=np.array([json_data['scene'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('environment', data=np.array([cur_task['environment'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('task_name', data=np.array([cur_task['task_name'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('task_name_candidates', 
                            data=np.array([s.encode('utf-8') for s in cur_task['task_name_candidates']], dtype='S'), 
                            dtype=string_dt)
    metadata.create_dataset('sample_rate', data=np.array([rate], dtype='S'), dtype=string_dt)
    metadata.create_dataset('num_steps', data=np.array([num_steps], dtype='S'), dtype=string_dt)
    metadata.create_dataset('robot_name', data=np.array([json_data['robot_name'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('robot_type', data=np.array([json_data['robot_type'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('robot_description', data=np.array([json_data['robot_description'].encode('utf-8')], dtype='S'), dtype=string_dt)
    metadata.create_dataset('robot_arm1_joints_state_dim', data=np.array([json_data['robot_arm1_joints_state_dim']]))
    metadata.create_dataset('robot_arm2_joints_state_dim', data=np.array([json_data['robot_arm2_joints_state_dim']]))
    metadata.create_dataset('robot_master_arm1_joints_state_dim', data=np.array([json_data['robot_master_arm1_joints_state_dim']]))
    metadata.create_dataset('robot_master_arm2_joints_state_dim', data=np.array([json_data['robot_master_arm2_joints_state_dim']]))
    metadata.create_dataset('robot_arm1_eef_state_dim', data=np.array([json_data['robot_arm1_eef_state_dim']]))
    metadata.create_dataset('robot_arm2_eef_state_dim', data=np.array([json_data['robot_arm2_eef_state_dim']]))
    metadata.create_dataset('robot_master_arm1_eef_state_dim', data=np.array([json_data['robot_master_arm1_eef_state_dim']]))
    metadata.create_dataset('robot_master_arm2_eef_state_dim', data=np.array([json_data['robot_master_arm2_eef_state_dim']]))
    metadata.create_dataset('robot_arm1_gripper_state_dim', data=np.array([json_data['robot_arm1_gripper_state_dim']]))
    metadata.create_dataset('robot_arm2_gripper_state_dim', data=np.array([json_data['robot_arm2_gripper_state_dim']]))
    metadata.create_dataset('robot_master_arm1_gripper_state_dim', data=np.array([json_data['robot_master_arm1_gripper_state_dim']]))
    metadata.create_dataset('robot_master_arm2_gripper_state_dim', data=np.array([json_data['robot_master_arm2_gripper_state_dim']]))
    metadata.create_dataset('robot_base_state_dim', data=np.array([json_data['robot_base_state_dim']]))
    metadata.create_dataset('robot_arm1_joints_action_dim', data=np.array([json_data['robot_arm1_joints_action_dim']]))
    metadata.create_dataset('robot_arm2_joints_action_dim', data=np.array([json_data['robot_arm2_joints_action_dim']]))
    metadata.create_dataset('robot_arm1_eef_action_dim', data=np.array([json_data['robot_arm1_eef_action_dim']]))
    metadata.create_dataset('robot_arm2_eef_action_dim', data=np.array([json_data['robot_arm2_eef_action_dim']]))
    metadata.create_dataset('robot_arm1_gripper_action_dim', data=np.array([json_data['robot_arm1_gripper_action_dim']]))
    metadata.create_dataset('robot_arm2_gripper_action_dim', data=np.array([json_data['robot_arm2_gripper_action_dim']]))
    metadata.create_dataset('robot_base_action_dim', data=np.array([json_data['robot_base_action_dim']]))
    metadata.create_dataset('camera1_rgb_resolution', data=np.array([json_data['camera1_rgb_resolution']]))
    metadata.create_dataset('camera2_rgb_resolution', data=np.array([json_data['camera2_rgb_resolution']]))
    metadata.create_dataset('camera3_rgb_resolution', data=np.array([json_data['camera3_rgb_resolution']]))
    metadata.create_dataset('camera4_rgb_resolution', data=np.array([json_data['camera5_rgb_resolution']]))
    metadata.create_dataset('camera1_depth_resolution', data=np.array([json_data['camera1_depth_resolution']]))
    metadata.create_dataset('camera2_depth_resolution', data=np.array([json_data['camera2_depth_resolution']]))
    metadata.create_dataset('camera3_depth_resolution', data=np.array([json_data['camera3_depth_resolution']]))
    metadata.create_dataset('camera4_depth_resolution', data=np.array([json_data['camera5_depth_resolution']]))

def load_json():
    global json_data
    bait_type_path = os.path.dirname(__file__)
    with open(os.path.join(bait_type_path, 'metadata.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    json_data = data

def format_string(s, delimiter=' '):
    parts = s.split(delimiter)
    formatted_string = ''.join(part.capitalize() for part in parts)
    return formatted_string

def async_save_data(q, args, user, rate, json_data, cur_task):
    while True:
        data = q.get()
        if data["type"] == "quit":
            return
        if data["type"] != "start":
            continue
        
        dataset_path = data["path"]
        cur_datetime = data["datetime"]
        data_size = 0

        data_dict = {
            # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
            '/observations/arm_joints_state': [],
            '/observations/arm_qvel_state': [],
            '/observations/arm_effort_state': [],
            '/observations/arm_eef_state': [],
            '/action': [],
            '/master_arm_eef_state': [],
            '/master_arm_joints_state': [],
            '/base_state': [],
            # '/base_action_t265': [],
        }

        # 相机字典  观察的图像
        if args.save_video:
            file_name_list = [cur_datetime, '_', format_string(json_data['dataset_name'], '_'), '_',
                            format_string(json_data['robot_name'], '_'), '_', 
                            format_string(json_data['scene'], ' '), '_',
                            format_string(cur_task['environment'], '_'), '_',
                            format_string(cur_task['task_name'], '_'), '_',
                            str(args.episode_idx)]
        
            file_name = ''.join(file_name_list)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            unsave_path = dataset_path
            dataset_path = os.path.join(dataset_path, file_name)

            camera_name = {"camera_3_rgb_front": "camera3_rgb",
                           "camera_1_rgb_left": "camera1_rgb",
                           "camera_2_rgb_right": "camera2_rgb",
                           "camera_5_rgb_head": "camera5_rgb",}
            
            writer = {}
            for key in camera_name.keys():
                save_path = dataset_path + "_" + camera_name[key] + ".mp4"
                if key.find('rgb') != -1:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
                    video_writer = cv2.VideoWriter(save_path, fourcc, args.frame_rate, (640, 480))
                    writer[key] = video_writer
        else:
            for cam_name in args.camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

        if args.use_depth_image:
            for depth_cam_name in args.depth_camera_names:
                data_dict[f'/observations/images_depth/{depth_cam_name}'] = []

        while True:
            data = q.get()
            if data["type"] == "data":
                # 往字典里面添值
                # Timestep返回的qpos，qvel,effort
                ts = data["timestep"]
                action = data["action"]
                action_ee = data["action_ee"]

                data_dict['/observations/arm_joints_state'].append(ts.observation['qpos'])
                data_dict['/observations/arm_qvel_state'].append(ts.observation['qvel'])
                data_dict['/observations/arm_effort_state'].append(ts.observation['effort'])
                data_dict['/observations/arm_eef_state'].append(ts.observation['ee'])

                # 实际发的action
                data_dict['/action'].append(action)
                data_dict['/master_arm_eef_state'].append(action_ee)
                data_dict['/master_arm_joints_state'].append(action)
                data_dict['/base_state'].append(ts.observation['base_vel'])

                # 相机数据
                # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
                if args.save_video:
                    for cam_name in args.camera_names:
                        frame = ts.observation['images'][cam_name]
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8)
                        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        writer[cam_name].write(rgb)
                else:
                    for cam_name in args.camera_names:
                        data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
                        
                if args.use_depth_image:
                    for depth_cam_name in args.depth_camera_names:
                        data_dict[f'/observations/images_depth/{depth_cam_name}'].append(ts.observation['images_depth'][depth_cam_name])
            elif data["type"] == "stop":
                data_size = data["num_steps"] - 1
            elif data["type"] == "unsave":
                if args.save_video:
                    for key in writer.keys():
                        writer[key].release()
                    
                    if os.path.exists(unsave_path):
                        shutil.rmtree(unsave_path)
                else:
                    data_dict = {}
                break
            elif data["type"] == "save":
                count = len(data_dict['/action'])
                if args.use_step:
                    if (count < cur_task['step_min'] or count > cur_task['step_max']):
                        print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" % args.max_timesteps)
                        data_dict = {}
                        break
                else:
                    if (count < cur_task['step_min'] or count > cur_task['step_max'] + 200):
                        print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" % int(cur_task['step_max'] + 200))
                        data_dict = {}
                        break 

                if args.save_video:
                    for key in writer.keys():
                        writer[key].release()

                t0 = time.time()
                with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
                    # 文本的属性：
                    # 1 是否仿真
                    # 2 图像是否压缩
                    root.attrs['sim'] = False
                    root.attrs['compress'] = False

                    # 创建一个新的组observations，观测状态组
                    # 图像组
                    obs = root.create_group('observations')

                    if not args.save_video:
                        image = obs.create_group('images')
                        for cam_name in args.camera_names:
                            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                                        chunks=(1, 480, 640, 3), compression='gzip')
                    if args.use_depth_image:
                        image_depth = obs.create_group('images_depth')
                        for depth_cam_name in args.depth_camera_names:
                            _ = image_depth.create_dataset(depth_cam_name, (data_size, 480, 640), dtype='uint16',
                                                        chunks=(1, 480, 640), compression='gzip')

                    _ = obs.create_dataset('arm_joints_state', (data_size, 14))
                    _ = obs.create_dataset('arm_qvel_state', (data_size, 14))
                    _ = obs.create_dataset('arm_effort_state', (data_size, 14))
                    _ = obs.create_dataset('arm_eef_state', (data_size, 14))
                    _ = root.create_dataset('action', (data_size, 14))
                    _ = root.create_dataset('master_arm_eef_state', (data_size, 14))
                    _ = root.create_dataset('master_arm_joints_state', (data_size, 14))
                    _ = root.create_dataset('base_state', (data_size, 2))

                    # data_dict write into h5py.File
                    for name, array in data_dict.items():  
                        root[name][...] = array

                    # save metadata
                    save_metadata(root, cur_datetime, user, rate, data_size, json_data, cur_task)
                
                print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)
                break


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.master_arm_left_pos_deque = None
        self.master_arm_right_pos_deque = None
        self.puppet_arm_left_pos_deque = None
        self.puppet_arm_right_pos_deque = None
        self.img_head_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_head_depth_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_head_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.img_head_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.master_arm_left_pos_deque = deque()
        self.master_arm_right_pos_deque = deque()
        self.puppet_arm_left_pos_deque = deque()
        self.puppet_arm_right_pos_deque = deque()
        self.robot_base_deque = deque()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or len(self.img_head_deque) == 0 or\
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or \
                                                len(self.img_front_depth_deque) == 0 or len(self.img_head_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), 
                              self.img_right_deque[-1].header.stamp.to_sec(), 
                              self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_head_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), 
                              self.img_right_depth_deque[-1].header.stamp.to_sec(), 
                              self.img_front_depth_deque[-1].header.stamp.to_sec(), 
                              self.img_head_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), 
                              self.img_right_deque[-1].header.stamp.to_sec(), 
                              self.img_front_deque[-1].header.stamp.to_sec(), 
                              self.img_head_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_head_deque) == 0 or self.img_head_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_pos_deque) == 0 or self.master_arm_left_pos_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_pos_deque) == 0 or self.master_arm_right_pos_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_pos_deque) == 0 or self.puppet_arm_left_pos_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_pos_deque) == 0 or self.puppet_arm_right_pos_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_head_depth_deque) == 0 or self.img_head_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        # print("img_left:", img_left.shape)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.img_head_deque[0].header.stamp.to_sec() < frame_time:
            self.img_head_deque.popleft()
        img_head = self.bridge.imgmsg_to_cv2(self.img_head_deque.popleft(), 'passthrough')

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        while self.master_arm_left_pos_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_pos_deque.popleft()
        master_arm_left_pos = self.master_arm_left_pos_deque.popleft()

        while self.master_arm_right_pos_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_pos_deque.popleft()
        master_arm_right_pos = self.master_arm_right_pos_deque.popleft()

        while self.puppet_arm_left_pos_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_pos_deque.popleft()
        puppet_arm_left_pos = self.puppet_arm_left_pos_deque.popleft()

        while self.puppet_arm_right_pos_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_pos_deque.popleft()
        puppet_arm_right_pos = self.puppet_arm_right_pos_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            # top, bottom, left, right = 40, 40, 0, 0
            # img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
        # top, bottom, left, right = 40, 40, 0, 0
        # img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')
        # top, bottom, left, right = 40, 40, 0, 0
        # img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_head_depth = None
        if self.args.use_depth_image:
            while self.img_head_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_head_depth_deque.popleft()
            img_head_depth = self.bridge.imgmsg_to_cv2(self.img_head_depth_deque.popleft(), 'passthrough')
        # top, bottom, left, right = 40, 40, 0, 0
        # img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_head, img_front, img_left, img_right, img_head_depth, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, 
                master_arm_left_pos, master_arm_right_pos, puppet_arm_left_pos, puppet_arm_right_pos, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 500:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 500:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 500:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_head_callback(self, msg):
        if len(self.img_head_deque) >= 500:
            self.img_head_deque.popleft()
        self.img_head_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 500:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 500:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 500:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)
    
    def img_head_depth_callback(self, msg):
        if len(self.img_head_depth_deque) >= 500:
            self.img_head_depth_deque.popleft()
        self.img_head_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 500:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 500:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 500:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 500:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def master_arm_left_pos_callback(self, msg):
        if len(self.master_arm_left_pos_deque) >= 500:
            self.master_arm_left_pos_deque.popleft()
        self.master_arm_left_pos_deque.append(msg)
    
    def master_arm_right_pos_callback(self, msg):
        if len(self.master_arm_right_pos_deque) >= 500:
            self.master_arm_right_pos_deque.popleft()
        self.master_arm_right_pos_deque.append(msg)

    def puppet_arm_left_pos_callback(self, msg):
        if len(self.puppet_arm_left_pos_deque) >= 500:
            self.puppet_arm_left_pos_deque.popleft()
        self.puppet_arm_left_pos_deque.append(msg)

    def puppet_arm_right_pos_callback(self, msg):
        if len(self.puppet_arm_right_pos_deque) >= 500:
            self.puppet_arm_right_pos_deque.popleft()
        self.puppet_arm_right_pos_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 500:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_head_topic, Image, self.img_head_callback, queue_size=500, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=500, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=500, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=500, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_head_depth_topic, Image, self.img_head_depth_callback, queue_size=500, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=500, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.master_arm_left_pos, PoseStamped, self.master_arm_left_pos_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_pos, PoseStamped, self.master_arm_right_pos_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_pos, PoseStamped, self.puppet_arm_left_pos_callback, queue_size=500, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_pos, PoseStamped, self.puppet_arm_right_pos_callback, queue_size=500, tcp_nodelay=True)

        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=500, tcp_nodelay=True)

    def process(self, queue):
        global key_pressed, num_steps
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0

        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True

        while not rospy.is_shutdown():
            # 2 收集数据
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            count += 1
            (img_head, img_front, img_left, img_right, img_head_depth, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, 
             master_arm_left_pos, master_arm_right_pos, puppet_arm_left_pos, puppet_arm_right_pos, robot_base) = result
            # 2.1 图像信息
            image_dict = dict()
            image_dict[self.args.camera_names[0]] = img_front
            image_dict[self.args.camera_names[1]] = img_left
            image_dict[self.args.camera_names[2]] = img_right
            image_dict[self.args.camera_names[3]] = img_head

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.args.depth_camera_names[0]] = img_front_depth
                image_dict_depth[self.args.depth_camera_names[1]] = img_left_depth
                image_dict_depth[self.args.depth_camera_names[2]] = img_right_depth
                image_dict_depth[self.args.depth_camera_names[3]] = img_head_depth
                obs['images_depth'] = image_dict_depth
            obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            obs['ee'] = np.array([puppet_arm_left_pos.pose.position.x, puppet_arm_left_pos.pose.position.y, puppet_arm_left_pos.pose.position.z, 
                                 puppet_arm_left_pos.pose.orientation.x, puppet_arm_left_pos.pose.orientation.y, puppet_arm_left_pos.pose.orientation.z, 
                                 puppet_arm_left_pos.pose.orientation.w,
                                 puppet_arm_right_pos.pose.position.x, puppet_arm_right_pos.pose.position.y, puppet_arm_right_pos.pose.position.z, 
                                 puppet_arm_right_pos.pose.orientation.x, puppet_arm_right_pos.pose.orientation.y, puppet_arm_right_pos.pose.orientation.z, 
                                 puppet_arm_right_pos.pose.orientation.w])
            if self.args.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            action = np.concatenate((np.array(master_arm_left.position), np.array(master_arm_right.position)), axis=0)
            action_pos = np.array([master_arm_left_pos.pose.position.x, master_arm_left_pos.pose.position.y, master_arm_left_pos.pose.position.z, 
                                 master_arm_left_pos.pose.orientation.x, master_arm_left_pos.pose.orientation.y, master_arm_left_pos.pose.orientation.z, 
                                 master_arm_left_pos.pose.orientation.w,
                                 master_arm_right_pos.pose.position.x, master_arm_right_pos.pose.position.y, master_arm_right_pos.pose.position.z, 
                                 master_arm_right_pos.pose.orientation.x, master_arm_right_pos.pose.orientation.y, master_arm_right_pos.pose.orientation.z, 
                                 master_arm_right_pos.pose.orientation.w])
            data = {}
            data["type"] = "data"
            data["action"] = action
            data["action_ee"] = action_pos
            data["timestep"] = ts
            queue.put(data)
            print("Frame data: ", count)
            if rospy.is_shutdown():
                exit(-1)

            if self.args.use_step:
                if count >= self.args.max_timesteps + 1:
                    num_steps = count
                    break 
            else:
                if key_pressed: 
                    key_pressed = None
                    num_steps = count
                    queue.put({"type": "stop", "num_steps": num_steps})
                    break

            rate.sleep()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--user', action='store', type=str, help='user.',
                        default="user", required=False)
    parser.add_argument('--task', action='store', type=str, help='Task.',
                        default="task", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['camera_3_rgb_front', 'camera_1_rgb_left', 'camera_2_rgb_right', 'camera_5_rgb_head'], required=False)
    parser.add_argument('--depth_camera_names', action='store', type=str, help='depth_camera_names',
                        default=['camera_3_depth_front', 'camera_1_depth_left', 'camera_2_depth_right', 'camera_5_depth_head'], required=False)
    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    parser.add_argument('--img_head_topic', action='store', type=str, help='img_head_topic',
                        default='/camera_t/color/image_raw', required=False)
    
    # topic name of depth image
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_rect_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_rect_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_rect_raw', required=False)
    parser.add_argument('--img_head_depth_topic', action='store', type=str, help='img_head_depth_topic',
                        default='/camera_t/depth/image_rect_raw', required=False)
    
    # topic name of arm
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    # topic name of arm pos
    parser.add_argument('--master_arm_left_pos', action='store', type=str, help='master_arm_left_pos',
                        default='/master/end_left', required=False)
    parser.add_argument('--master_arm_right_pos', action='store', type=str, help='master_arm_right_pos',
                        default='/master/end_right', required=False)
    parser.add_argument('--puppet_arm_left_pos', action='store', type=str, help='puppet_arm_left_pos',
                        default='/puppet/end_left', required=False)
    parser.add_argument('--puppet_arm_right_pos', action='store', type=str, help='puppet_arm_right_pos',
                        default='/puppet/end_right', required=False)
    
    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=True, required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)

    parser.add_argument('--use_step', action='store', type=bool, help='use_step',
                        default=False, required=False)
    
    parser.add_argument('--use_async', action='store', type=int, help='use_async',
                        default=True, required=False)
    parser.add_argument('--process_count', action='store', type=int, help='process_count',
                        default=3, required=False)
    
    parser.add_argument('--save_video', action='store', type=bool, help='use_step',
                        default=True, required=False)
    
    args = parser.parse_args()
    return args


def main():
    global cur_datetime, user, rate, num_steps, on_process, json_data, cur_task
    args = get_arguments()

    if not args.use_step:
        stop_event = Event()  
        key_capture_thread = Thread(target=capture_key_thread, args=(stop_event,))  
        key_capture_thread.start()  
        global key_pressed

    ros_operator = RosOperator(args)

    load_json()
    input_metadata_info(args)
    
    if args.use_async:
        queue_list = list()
        process_list = list()
        for i in range(args.process_count):
            q = Queue(500)
            pro = multiprocessing.Process(target=async_save_data, args=(q, 
                                                                        copy.deepcopy(args), 
                                                                        copy.deepcopy(user), 
                                                                        copy.deepcopy(rate), 
                                                                        copy.deepcopy(json_data), 
                                                                        copy.deepcopy(cur_task)))
            pro.start()
            queue_list.append(q)
            process_list.append(pro)

    while True:
        # 系统空闲内存
        mem = psutil.virtual_memory()
        kx = float(mem.free) / 1024 / 1024 / 1024

        print('当前用户: %s'% user)
        print('当前环境: %s'% cur_task['environment'])
        print('当前任务: %s'% cur_task['task_name'])
        print('当前保存路径: %s' %args.dataset_dir)
        print('episode_idx: %d' %args.episode_idx)
        print('系统空闲内存: %d.3GB' %kx)
        print('CPU使用率: %f.2' %psutil.cpu_percent())

        if not args.use_step:
            print('手动停止模式：按任意键+回车停止')
        
        status = input("按回车键开始, q退出\n")
        if status == 'q' or status == 'Q':
            print('正在退出...')
            if args.use_async:
                print('等待保存完成')
                for q in queue_list:
                    q.put({"type": "quit"})
                for pro in process_list:
                    pro.join()
            print('退出')
            break
        
        on_process = True     
        dataset_dir = os.path.join(args.dataset_dir, user, args.task_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        cur_datetime = str(get_date_time())
        dataset_path = os.path.join(dataset_dir, cur_task['task_name'] + "_" + cur_datetime + "_episode_" + str(args.episode_idx))

        cur = args.episode_idx % args.process_count
        queue_list[cur].put({"type": "start", 
                             "path": dataset_path,
                             "datetime": cur_datetime})
        ros_operator.process(queue_list[cur])
        on_process = False

        status = input("采集结束是否保存(y/n)?\n")
        while status.lower() != 'y' and status.lower() != 'n':
            status = input("输入错误，确认是否保存(y/n)?\n")

        if status.lower() == 'y':
            print('保存中...')
            if args.use_async:
                queue_list[cur].put({"type": "save"})

            args.episode_idx = args.episode_idx + 1
        else :
            queue_list[cur].put({"type": "unsave"})
            print("不保存")

    if not args.use_step:
        stop_event.set()  # 停止按键捕获线程  
        key_capture_thread.join()  # 等待线程结束  

if __name__ == '__main__':
    main()

# python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 0
