from prismatic.vla.datasets import EpisodicRLDSDataset
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding

from ainnorobot_data.schema.episode_dataclass import Episode, Metadata, Observation, Step
import numpy as np
from PIL import Image
# from IPython import display

import re
import os
import cv2
import json
import pdb


def mkdir_ifmissing(fld):
    if not os.path.isdir(fld):
        os.makedirs(fld)

def save_video(array, filename='output_video.mp4', fps=10):
    if len(array.shape) != 4 or array.shape[3] != 3:
        raise ValueError('Input array must have shape [frames, height, width, 3]')

    frames, height, width, channels = array.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    # print(filename)

    for i in range(frames):
        frame = array[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # video_writer.write(frame)
        video_writer.write(rgb)

    video_writer.release()


import re

def capitalize_after_underscore(s):
    # 正则表达式匹配下划线后的第一个字符
    pattern = r'_([a-zA-Z])'
    
    # 使用 re.sub() 函数进行替换
    # \1 是一个特殊变量，它代表正则表达式中第一个括号内匹配到的内容
    # 在括号内的内容前面加上大写函数 str.upper() 来实现大写
    result = re.sub(pattern, lambda match: match.group(1).upper(), s)
    result2 = result.replace(' ','')

    
    return result2

def get_metadata_dim(state_encoding_type, action_ecoding_type, state_encoding_dict,action_ecoding_dict):
    robot_type = "single_arm"
    robot_arm1_eef_state_dim = -1
    robot_arm1_eef_action_dim = -1
    robot_arm1_joints_state_dim = -1
    robot_arm1_joints_action_dim = -1


    robot_arm2_eef_state_dim = -1
    robot_arm2_eef_action_dim = -1
    robot_arm2_joints_state_dim = -1
    robot_arm2_joints_action_dim = -1
    if state_encoding_type == 'joint_12_2':
        robot_arm1_joints_state_dim = 6
        robot_arm2_joints_state_dim = 6
        robot_type = 'dual_arm'
    else:
        if state_encoding_type == 'eef_6_1':
            robot_arm1_eef_state_dim = 6
        elif state_encoding_type == 'eef_7_1':
            robot_arm1_eef_state_dim =7
        elif state_encoding_type == 'joint_7_1':
            robot_arm1_joints_state_dim =7

    if action_ecoding_type == 'joint_12_2':
        robot_arm1_joints_action_dim = 6
        robot_arm2_joints_action_dim = 6
        robot_type = 'dual_arm'
    else:
        if action_ecoding_type == 'joint_7_1':
            robot_arm1_joints_action_dim =7
        elif action_ecoding_type == 'eef_6_1':
            robot_arm1_eef_action_dim = 6
        elif action_ecoding_type == 'eef_9_1':
            robot_arm1_eef_action_dim = 9

    metadata_dim_dict={
    'robot_arm1_eef_state_dim':robot_arm1_eef_state_dim,
    'robot_arm1_eef_action_dim': robot_arm1_eef_action_dim,
    'robot_arm1_joints_state_dim': robot_arm1_joints_state_dim,
    'robot_arm1_joints_action_dim': robot_arm1_joints_action_dim,
    'robot_arm2_eef_state_dim':robot_arm2_eef_state_dim,
    'robot_arm2_eef_action_dim': robot_arm2_eef_action_dim,
    'robot_arm2_joints_state_dim': robot_arm2_joints_state_dim,
    'robot_arm2_joints_action_dim': robot_arm2_joints_action_dim,
    'robot_type': robot_type

                    }

    return metadata_dim_dict




data_root_dir = "/mnt/nas03/ChatRobotDatasets/openx/dataset/"
output =  "/mnt/nas03/ChatRobotDatasets/openx/dataset-0812-tmp/"
mkdir_ifmissing(output)

error_names = ['berkeley_mvp_converted_externally_to_rlds',"berkeley_rpt_converted_externally_to_rlds"]

for key in OXE_DATASET_CONFIGS.keys():
    import pdb
    pdb.set_trace()
# for key in ['berkeley_mvp_converted_externally_to_rlds']:
    dataset_dir = os.path.join(data_root_dir,key)
    if os.path.exists(dataset_dir) and key not in error_names:

        # data_mix = "kuka"
        print("######################################")
        print(key)

        data_mix = key
        dataset_name = data_mix
        output_folder =os.path.join(output,dataset_name)
        data_config = OXE_DATASET_CONFIGS[dataset_name]

        mkdir_ifmissing(output_folder)



        '''
        # Defines Proprioceptive State Encoding Schemes
        class StateEncoding(IntEnum):
            # fmt: off
            NONE = -1               # No Proprioceptive State
            POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
            POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
            JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
            JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
            # fmt: on


        # Defines Action Encoding Schemes
        class ActionEncoding(IntEnum):
            # fmt: off
            EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
            JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
            JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
            EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
        '''


        state_encoding_dict = {1: 'eef_6_1',
                               2: 'eef_7_1',
                               3: 'joint_7_1',
                               4: 'joint_12_2',
                               -1: None}
        action_ecoding_dict = {1: 'eef_6_1',
                               2: 'joint_7_1',
                               3: 'joint_12_2',
                               4: 'eef_9_1'}


        state_encoding_num = int(data_config['state_encoding'].value)
        action_ecoding_num = int(data_config['action_encoding'].value)

        state_encoding_type = state_encoding_dict[state_encoding_num]
        action_ecoding_type = action_ecoding_dict[action_ecoding_num]
        image_keys = {"primary": 'camera4_rgb', 
                    "secondary": 'camera3_rgb',
                     "wrist": 'camera1_rgb'}
        ainnoneed_observation_keys = ["primary", "secondary", "wrist",
                                       'proprio' ]

        # resolution = get_orig_resolution(dataset_dir, data_config)
        try:

            cls = EpisodicRLDSDataset
            dataset = cls(
                data_root_dir,
                data_mix,
                # resize_resolution=default_image_resolution[1:],
                # resize_resolution = (224,224),
                # shuffle_buffer_size=shuffle_buffer_size,
                train=False,
                image_aug=False,
            )
            # 

            cam_images_dict={'camera1_rgb':[],
                             'camera2_rgb':[], 
                             'camera3_rgb':[],
                             'camera4_rgb':[],}
            cam_resolution_dict={'camera1_rgb':[],
                             'camera2_rgb':[], 
                             'camera3_rgb':[],
                             'camera4_rgb':[],}

            #加入循环
            # pdb.set_trace()

            episode_id = 0
            for idx, episode_data in enumerate(dataset.dataset.iterator()):
                # episode_data=next(dataset.dataset.iterator())
                print(idx)
                observation = episode_data['observation']
                action_data = episode_data['action']


                import pdb

                for key in observation.keys(): 

                     #dict_keys(['image_primary', 'proprio', 'timestep', 'pad_mask_dict', 'pad_mask'])
                    if key.split('_')[-1] in image_keys:
                        rename_cam = image_keys[key.split('_')[-1]]
                        cam_images_array = observation[key]
                        cam_images_dict[rename_cam].append(cam_images_array)
                    elif key == 'proprio':
                        obs_states = observation[key]

                step_datas_list=[]
                # print(len(action_data))
                for step_idx in range(len(action_data)):
                    arm1_joints_state = []
                    arm1_eef_state = []
                    arm1_joints_action = []
                    arm1_eef_action = []

                    arm2_joints_state = []
                    arm2_eef_state = []
                    arm2_joints_action = []
                    arm2_eef_action = []

                    arm1_gripper_state = []
                    arm2_gripper_state=[]
                    arm1_gripper_action = []
                    arm2_gripper_action = []
                    # print(step_idx)
                    # pdb.set_trace()

                    try:
                        # print(step_idx)
                        lang_instruction = episode_data['task']['language_instruction'][step_idx]
                        step_state = obs_states[step_idx][0].tolist()
                        step_action = action_data[step_idx][0].tolist()
                        # pdb.set_trace()
                        if state_encoding_type !=None:
                            if state_encoding_type != 'joint_12_2':
                                if state_encoding_type.startswith('eef'):
                                    arm1_eef_state = step_state
                                    arm1_gripper_state = [arm1_eef_state.pop()]

                                else:
                                    arm1_joints_state = step_state
                                    arm1_gripper_state = [arm1_joints_state.pop()]
                            else:
                                arm1_joints_state = step_state[:len(step_state)//2]
                                arm1_gripper_state = arm1_joints_state.pop()
                                arm2_joints_state = step_state[len(step_state)//2:]
                                arm2_gripper_state = arm2_joints_state.pop()

                        if action_ecoding_type != 'joint_12_2':
                            if action_ecoding_type.startswith('eef'):
                                arm1_eef_action = step_action
                                arm1_gripper_action = [arm1_eef_action.pop()]
                            else:
                                arm1_joints_action = step_action
                                arm1_gripper_action = [arm1_joints_action.pop()]
                        else:
                            arm1_joints_action = step_action[:len(step_action)//2]
                            arm1_gripper_action = [arm1_joints_action.pop()]
                            arm2_joints_action = step_state[len(step_action)//2:]
                            arm2_gripper_action = [arm2_joints_action.pop()]

                        step_data = Step(
                                  observation=Observation(
                                      lang_instruction=lang_instruction, 
                                      arm1_eef_state=arm1_eef_state,
                                      arm1_gripper_state=arm1_gripper_state,
                                      arm1_joints_state = arm1_joints_state,
                                      arm2_eef_state=arm2_eef_state,
                                      arm2_gripper_state=arm2_gripper_state,
                                      arm2_joints_state = arm2_joints_state,
                                      ),
                                  arm1_eef_action=arm1_eef_action,
                                  arm1_gripper_action=arm1_gripper_action,
                                  arm1_joints_action = arm1_joints_action,

                                  arm2_eef_action=arm2_eef_action,
                                  arm2_gripper_action=arm2_gripper_action,
                                  arm2_joints_action = arm2_joints_action,

                                  )
                        step_datas_list.append(step_data)
                        # print('成功')

                    except IOError:
                        print('step data失败一次')


                scene = 'Null'
                task_name = 'Null'
                environment_name = 'Null'
                robot_name = dataset_name
                experiment_time = '20240812170000'
                metadata_dim_dict = get_metadata_dim(state_encoding_type, action_ecoding_type,
                                                            state_encoding_dict,action_ecoding_dict)
                episode_id +=1
                task_name_string = capitalize_after_underscore(task_name)
                dataset_name_string = capitalize_after_underscore(dataset_name)
                robot_name_string = capitalize_after_underscore(robot_name)

                datafile_name=experiment_time+'_'+dataset_name_string+'_'+robot_name_string+'_'+scene+'_'+environment_name+'_'+task_name_string+'_'+str(episode_id) 
                #存视频
                for key in cam_images_dict.keys():
                    cam_images = cam_images_dict[key]
                    if len(cam_images)!= 0:
                        images_array = np.squeeze(cam_images[0])

                        cam_name = key
                        cam_filename = os.path.join(output_folder, datafile_name+'_'+cam_name+'.mp4')
                        # print(cam_filename)
                        if len(images_array.shape) != 4 or images_array.shape[3] != 3:
                            print('Input array must have shape [frames, height, width, 3]')
                        else:
                            num_step = images_array.shape[0]
                            image_shape = images_array.shape[1:3]
                            save_video(images_array, cam_filename)
                            cam_resolution_dict[key]=image_shape


                episode_json = Episode(
                                        metadata=Metadata(
                                                              dataset_name=dataset_name,
                                                              episode_id=episode_id,
                                                              scene=scene,
                                                              environment=environment_name,
                                                              task_name=task_name,
                                                              num_steps=len(step_datas_list),
                                                              robot_name=robot_name,
                                                              robot_type=metadata_dim_dict['robot_type'],
                                                              # robot_description=robot_description,
                                                            robot_arm1_eef_state_dim = metadata_dim_dict['robot_arm1_eef_state_dim'],
                                                            robot_arm1_eef_action_dim = metadata_dim_dict ['robot_arm1_eef_action_dim'],
                                                            robot_arm1_joints_state_dim = metadata_dim_dict ['robot_arm1_joints_state_dim'],
                                                            robot_arm1_joints_action_dim=  metadata_dim_dict['robot_arm1_joints_action_dim'],
                                                            robot_arm2_eef_state_dim = metadata_dim_dict['robot_arm2_eef_state_dim'],
                                                            robot_arm2_eef_action_dim = metadata_dim_dict['robot_arm2_eef_action_dim'],
                                                            robot_arm2_joints_state_dim= metadata_dim_dict['robot_arm2_joints_state_dim'],
                                                            robot_arm2_joints_action_dim = metadata_dim_dict['robot_arm2_joints_action_dim'],


                                                              robot_arm1_gripper_state_dim=1,
                                                              robot_arm1_gripper_action_dim=1,

                                                              camera1_rgb_resolution = cam_resolution_dict['camera1_rgb'],   #[h,w]
                                                              camera2_rgb_resolution = cam_resolution_dict['camera2_rgb'],
                                                              camera3_rgb_resolution = cam_resolution_dict['camera3_rgb'],
                                                              camera4_rgb_resolution = cam_resolution_dict['camera4_rgb'],

                                                              experiment_time = experiment_time,
                                                              # operator =operator,
                                                              # task_name_candidates = task_name_candidates
                                                              ),
                                        steps=step_datas_list
                                             )

                jsonstr = episode_json.dict()
                  # write json file and video file
                # pdb.set_trace()
                with open(f"{output_folder}/{datafile_name}.json", "w",encoding='utf-8') as f:
                    json.dump(jsonstr, f,ensure_ascii=False,indent=2)


        except IOError:
            print('data失败一次')






