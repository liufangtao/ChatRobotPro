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


def mkdir_ifmissing(fld):
    if not os.path.isdir(fld):
        os.makedirs(fld)

def load_json(json_path):
    with open(json_path, 'r',encoding='utf-8') as file:
        data=json.load(file)
    return data

def save_video2img(video_path):
    cap = VideoCapture(video_path)
    if not cap.isOpened():
        print('无法打开视频文件')
        exit()

    image_array = []

    ret = True
    num = 1
    is_all_frame = True
    time_interval = 1
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
    metadata=root.create_group('metadata')
    task_name = data_json["metadata"]["task_name"]
    sample_rate = data_json["metadata"]["sample_rate"]

    metadata.create_dataset('task_name',data=np.array([task_name],dtype='S'),dtype=string_dt)
    metadata.create_dataset('sample_rate',data=np.array([sample_rate],dtype='S'),dtype=string_dt)
    
# 保存数据函数
def save_data(episode_datadict,dataset_path):
    # 数据字典
    camera_rgb_names = ['camera_3_rgb_front', 'camera_1_rgb_left', 'camera_2_rgb_right', 'camera_5_rgb_head']
    camera_depth_names = ['camera_3_depth_front', 'camera_1_depth_left', 'camera_2_depth_right', 'camera_5_depth_head']
    data_size = len(episode_datadict['observations']['arm_joints_state'])
    data_dict = episode_datadict

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_rgb_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), compression='gzip',
                                    data = data_dict['observations']['images'][cam_name])
        if use_depthimg_flag:
            pdb.set_trace()
            image_depth = obs.create_group('images_depth')
            for depth_cam_name in camera_depth_names:
                _ = image_depth.create_dataset(depth_cam_name, (data_size, 480, 640), dtype='uint16', chunks=(1, 480, 640), compression='gzip',
                                        data=data_dict['observations']['images_depth'][depth_cam_name])
        
        _ = obs.create_dataset('arm_joints_state', (data_size, 14),data=np.array(data_dict['observations']['arm_joints_state']))
        _ = obs.create_dataset('arm_eef_state', (data_size, 14), data=np.array(data_dict['observations']['arm_eef_state']))
        _ = root.create_dataset('base_state', (data_size, 2), data=np.array(data_dict['observations']['base_state']))
        root.close()
        # # data_dict write into h5py.File
        # for name, array in data_dict.items():  
        #     root[name][...] = array
    
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)

def create_datadict(episode_datainfo,ainno_data_folder):
    json_path = os.path.join(ainno_data_folder,episode_datainfo['jsonfile'][0])
    for rgb_file in episode_datainfo['cam_rgb_files']:
        if 'camera5' in rgb_file:
            camera_5_rgb_path = os.path.join(ainno_data_folder,rgb_file)
        elif 'camera3' in rgb_file:
            camera_3_rgb_path = os.path.join(ainno_data_folder,rgb_file)
        elif 'camera1' in rgb_file:
            camera_1_rgb_path = os.path.join(ainno_data_folder,rgb_file)
        elif 'camera2' in rgb_file:
            camera_2_rgb_path = os.path.join(ainno_data_folder,rgb_file)

    for rgb_file in episode_datainfo['cam_depth_files']:
        if 'camera5' in rgb_file:
            camera_5_depth_path = os.path.join(ainno_data_folder,rgb_file)
        elif 'camera3' in rgb_file:
            camera_3_depth_path = os.path.join(ainno_data_folder,rgb_file)
        elif 'camera1' in rgb_file:
            camera_1_depth_path = os.path.join(ainno_data_folder,rgb_file)
        elif 'camera2' in rgb_file:
            camera_2_depth_path = os.path.join(ainno_data_folder,rgb_file)


    data_json =load_json(json_path)
    task_name = data_json["metadata"]["task_name"]
    episode_id = data_json["metadata"]["episode_id"]

    # data_dict= {

    #         '/observations/arm_joints_state': [],
    #         '/observations/arm_eef_state': [],
    #         '/observations/images': {},
    #         '/observations/images_depth': {},

    #         '/base_state': [],

    #     }

    data_dict= {
            'observations':{'images':{},
                            'images_depth':{},
                            'arm_joints_state':[],
                            "arm_eef_state":[]},
            'base_state':[]


        }


    step_datas=data_json['steps']
    observation_joints_list = []
    observation_eef_list = []
    base_state_list = []
    data_dict['observations']['images']['camera_5_rgb_head'] = save_video2img(camera_5_rgb_path)
    data_dict['observations']['images']['camera_1_rgb_left'] = save_video2img(camera_1_rgb_path)
    data_dict['observations']['images']['camera_2_rgb_right'] = save_video2img(camera_2_rgb_path)
    data_dict['observations']['images']['camera_3_rgb_front'] = save_video2img(camera_3_rgb_path)

    data_dict['observations']['images_depth']['camera_5_depth_head'] = save_video2img(camera_5_depth_path)
    data_dict['observations']['images_depth']['camera_1_depth_left'] = save_video2img(camera_1_depth_path)
    data_dict['observations']['images_depth']['camera_2_depth_right'] = save_video2img(camera_2_depth_path)
    data_dict['observations']['images_depth']['camera_3_depth_front'] = save_video2img(camera_3_depth_path)


    for idx, step in enumerate(step_datas):
        import pdb
        # pdb.set_trace()
        arm1_joints_state = step['observation']['arm1_joints_state']
        arm2_joints_state = step['observation']['arm2_joints_state']
        arm1_eef_state = step['observation']['arm1_eef_state']
        arm2_eef_state = step['observation']['arm2_eef_state']
        arm1_gripper_state = [i*5 for i in step['observation']["arm1_gripper_state"]]
        arm2_gripper_state = [i*5 for i in step['observation']["arm2_gripper_state"]]
        joints_state = arm1_joints_state+arm1_gripper_state+arm2_joints_state+arm2_gripper_state
        eef_state = arm1_eef_state+arm1_gripper_state+arm2_eef_state+arm2_gripper_state

        observation_joints_list.append(joints_state)
        observation_eef_list.append(eef_state)

        base_state = step['observation']["base_state"]
        base_state_list.append(base_state)

    data_dict['observations']['base_state']=base_state_list
    data_dict['observations']['arm_joints_state'] = observation_joints_list
    data_dict['observations']['arm_eef_state'] = observation_eef_list
    return data_dict, episode_id

ainno_data_folder = "/mnt/nas03/cobot_magic_datasets/data20240731-ainno-addcam"
ouput_cobot_folder = ainno_data_folder+'-back4train'
mkdir_ifmissing(ouput_cobot_folder)
use_depthimg_flag = 0
#遍历文件夹做文件名匹配
json2data_dict = {}
import pdb
# pdb.set_trace()
for file in os.listdir(ainno_data_folder):
    filepath = os.path.join(ainno_data_folder,file)
    if os.path.isfile(filepath):
        if file.endswith('.json'):
            filehead = file.split('.json')[0]
            if filehead not in json2data_dict.keys():
                json2data_dict[filehead] = {'jsonfile':[],
                                            'cam_rgb_files':[],
                                            'cam_depth_files':[]}
            json2data_dict[filehead]['jsonfile'].append(file)
        elif file.endswith('.mp4'):
            filehead = file.split('_camera')[0]
            if filehead not in json2data_dict.keys():
                json2data_dict[filehead] = {'jsonfile':[],
                                            'cam_rgb_files':[],
                                            'cam_depth_files':[]}
            cam_type = file.split('.')[0][-3:]
            if cam_type == 'rgb':
                json2data_dict[filehead]['cam_rgb_files'].append(file)
            else:
                json2data_dict[filehead]['cam_depth_files'].append(file)

for filehead in json2data_dict.keys():
    episode_datainfo = json2data_dict[filehead]
    if len(episode_datainfo['jsonfile'])==1 and len(episode_datainfo['cam_rgb_files'])==4 and len(episode_datainfo['cam_depth_files'])==4:
        episode_datadict, _ = create_datadict(episode_datainfo,ainno_data_folder)
        output_path = os.path.join(ouput_cobot_folder,filehead)
        # pdb.set_trace()
        print(output_path)

        save_data(episode_datadict,output_path)











