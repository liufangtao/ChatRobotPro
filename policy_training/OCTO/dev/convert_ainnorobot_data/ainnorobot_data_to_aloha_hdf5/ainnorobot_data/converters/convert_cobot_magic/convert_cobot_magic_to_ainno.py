import re
import cv2
import numpy as np
import h5py
from PIL import Image
import os
import json
# from schema.episode_dataclass import Episode, Metadata, Observation, Step
from ainnorobot_data.schema.episode_dataclass import Episode, Metadata, Observation, Step
# TODO schema add cam and revise cam No



def mkdir_ifmissing(fld):
    if not os.path.isdir(fld):
        os.makedirs(fld)

def normalize_gripper_value(gripper_value, max_value=5):
    gripper_normalized = 0
    if gripper_value<0:
        gripper_normalized=0
    elif gripper_value>max_value:
        gripper_normalized=1
    else:
        gripper_normalized = gripper_value/max_value
    return gripper_normalized


def save_video(array, filename='output_video.mp4', fps=30):
    if len(array.shape) != 4 or array.shape[3] != 3:
        raise ValueError('Input array must have shape [frames, height, width, 3]')
    frames, height, width, channels = array.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    print(filename)

    for i in range(frames):
        frame = array[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # video_writer.write(frame)
        video_writer.write(rgb)

    video_writer.release()


def save_depth_video(frames, output_video_path, frame_rate=30.0, is_color=False, codec='XVID'):
    frame_height, frame_width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height), isColor=is_color)
 
    for frame in frames:
        out.write(frame)
 
    out.release()
    print(f"Video saved as {output_video_path}")

# def save_depth_video(array, filename='output_video.mp4', fps=30):
#     # if len(array.shape) != 4 or array.shape[3] != 3:
#     #     raise ValueError('Input array must have shape [frames, height, width, 3]')
#     frames, height, width = array.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
#     video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
#     print(filename)

#     for i in range(frames):
#         frame = array[i]
#         if frame.dtype != np.uint8:
#             frame = (frame * 255).astype(np.uint8)
#         # rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         video_writer.write(frame)
#         # video_writer.write(rgb)

#     video_writer.release()
# def encode2b(task_names_chinese):
#     task_name_candidates=[]
#     for task_name_str in task_names_chinese:
#         ascii_byte_string=task_name_str.encode('utf-8')
#         task_name_candidates.append(ascii_byte_string)
#     return task_name_candidates

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

import re

def process_string(s):
    # 正则表达式匹配下划线后的第一个字符，并将其大写
    s=s.replace(' ','_')
    s = re.sub(r'(?<=_)([a-zA-Z])', lambda match: match.group(1).upper(), s)
    # 去除字符串中的所有下划线
    s = s.replace('_', '')
    # s = s.replace(' ','')
    # 将字符串首字母大写
    s = s[0].upper() + s[1:]
    return s

def covert_cobot_to_ainno(data_folder,output_folder,start_id):
    episode_id = start_id-1
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for episode_name in filenames:
            data_path = os.path.join(dirpath,episode_name)
            if episode_name.endswith('.hdf5'):
                print(episode_name)
                episode_head = episode_name.split('.')[0]   #TODO episode_name待解析
                # data_path = os.path.join(data_folder,episode_name)
                try:
                    print(data_path)
                    episode_data=h5py.File(data_path,'r')   #<KeysViewHDF5 ['action', 'base_state', 'master_arm_eef_state', 'master_arm_joints_state', 'metadata', 'observations']
                    episode_id += 1
                    #初始化
                    import pdb
                    # pdb.set_trace()
                    robot_name ='cobot_magic'
                    robot_name=process_string(robot_name)
                    dataset_name="cobot_magic"
                    dataset_name=process_string(dataset_name)
                    exp_time = episode_data['metadata']['experiment_time'][0].decode('utf-8')
                    operator =episode_data['metadata']['operator'][0].decode('utf-8')  #TODO 根据数据组织待修改
                    # operator = process_string(operator)
                    # task_name_candidates = ['左臂抓取桌面上的可乐，传递给右臂，右臂把可乐放置在桌面中间的固定点',
                    #                         '左臂从桌面上拿起可乐，然后传递给右臂，右臂再将可乐放到桌面中间的固定位置',
                    #                         '左臂抓取桌面上的可乐后，转交给右臂，右臂负责将可乐放置在桌面中间的固定点',
                    #                         '左臂从桌面上拾起可乐，传递给右臂，右臂随后把可乐放置在桌面中央的固定位置'
                    #                         ]   #TODO
                    task_name_candidates = episode_data['metadata']['task_name_candidates']
                    task_name_candidates = [t.decode('utf-8') for t in task_name_candidates]

                    sample_rate = episode_data['metadata']['sample_rate'][0].decode('utf-8')
                    scene = episode_data['metadata']['scene'][0].decode('utf-8')
                    scene=process_string(scene)

                    environment_name=episode_data['metadata']['environment'][0].decode('utf-8')
                    environment_name = process_string(environment_name)
                    task_name =episode_data['metadata']['task_name'][0].decode('utf-8')
                    task_name = process_string(task_name)
                    robot_description = episode_data['metadata']['robot_description'][0].decode('utf-8')

                    # pdb.set_trace()

                    terminal_flag = False
                    step_datas_list = []
                    reward = 0.0
                    discount = 0.0

                    actions = episode_data['action']
                    # base_actions = episode_data['base_action']
                    num_steps = actions.shape[0]

                    obsevations=episode_data['observations']  # ['arm_eef_state', 'arm_effort_state', 'arm_joints_state', 'arm_qvel_state', 'images', 'images_depth']
                    images_data = obsevations['images']  
                    # cam_images_high = images_data['camera_4_rgb_head']   #TODO['camera_1_rgb_left', 'camera_2_rgb_right', 'camera_3_rgb_front', 'camera_4_rgb_head']
                    cam_images_head = images_data['camera_5_rgb_head']
                    cam_images_left = images_data['camera_1_rgb_left']
                    cam_images_right = images_data['camera_2_rgb_right']
                    cam_images_front = images_data['camera_3_rgb_front']

                    depthimages_data = obsevations['images_depth']   
                    depthcam_images_head = depthimages_data['camera_5_depth_head']   
                    depthcam_images_left = depthimages_data['camera_1_depth_left']
                    depthcam_images_right = depthimages_data['camera_2_depth_right']
                    depthcam_images_front = depthimages_data['camera_3_depth_front']

                    # assert images_data.shape[0]==actions.shape[0]
                    steps = actions.shape[0]
                    for step_idx in range(steps):
                        # print(step_idx)

                        action = actions[step_idx]
                        action_single_dim = len(action)//2
                        arm1_joints_action =action[:action_single_dim].tolist()  #左臂
                        arm2_joints_action =action[action_single_dim:].tolist()  #右臂
                        arm1_gripper_action = [normalize_gripper_value(arm1_joints_action.pop())]
                        arm2_gripper_action = [normalize_gripper_value(arm2_joints_action.pop())]
                        # base_action = base_actions[step_idx].tolist()
                        base_state = episode_data['base_state'][step_idx].tolist()
                        #observation
                        # cam_img_high = cam_images_high[step_idx]
                        # cam_img_left = cam_images_left[step_idx]
                        # cam_img_right = cam_images_right[step_idx]
                        if 'master_arm_eef_state' in episode_data:
                            master_eef_states = episode_data['master_arm_eef_state'][step_idx].tolist()
                            master_arm1_eef_state = master_eef_states[:len(master_eef_states)//2]
                            master_arm2_eef_state = master_eef_states[len(master_eef_states)//2:]
                            master_arm1_gripper_state_value = master_arm1_eef_state.pop()
                            master_arm2_gripper_state_value = master_arm2_eef_state.pop()
                            master_arm1_gripper_state = [normalize_gripper_value(master_arm1_gripper_state_value)]
                            master_arm2_gripper_state = [normalize_gripper_value(master_arm2_gripper_state_value)]
                        if 'master_arm_joints_state' in episode_data:
                            master_joints_states = episode_data['master_arm_joints_state'][step_idx].tolist()
                            master_arm1_joints_state = master_joints_states[:len(master_joints_states)//2]
                            master_arm2_joints_state = master_joints_states[len(master_joints_states)//2:]
                            master_arm1_gripper_state = master_arm1_joints_state.pop()
                            master_arm2_gripper_state = master_arm2_joints_state.pop()
                            master_arm1_gripper_state = [normalize_gripper_value(master_arm1_gripper_state_value)]
                            master_arm2_gripper_state = [normalize_gripper_value(master_arm2_gripper_state_value)]
                        # pdb.set_trace()

                        if 'arm_joints_state' in obsevations:
                            qpos = obsevations['arm_joints_state'][step_idx].tolist()
                            arm1_joints_state = qpos[:len(qpos)//2]
                            arm2_joints_state = qpos[len(qpos)//2:]
                            arm1_gripper_state = [normalize_gripper_value(arm1_joints_state.pop())]
                            arm2_gripper_state = [normalize_gripper_value(arm2_joints_state.pop())]

                            if step_idx < steps-1:
                                joints_state_next = obsevations['arm_joints_state'][step_idx+1].tolist()
                                arm1_joints_state_next = joints_state_next[:len(joints_state_next)//2][0:-1]
                                arm2_joints_state_next = joints_state_next[len(joints_state_next)//2:][0:-1]
                                # pdb.set_trace()
                                arm1_joints_action = [a-b for a,b in zip(arm1_joints_state_next,arm1_joints_state)]  #delta
                                arm2_joints_action = [a-b for a,b in zip(arm2_joints_state_next,arm2_joints_state)]
                            else:
                                arm1_joints_action = [0]*6  
                                arm2_joints_action = [0]*6  
                        if 'arm_eef_state' in obsevations:
                            eef_state = obsevations['arm_eef_state'][step_idx].tolist()
                            arm1_eef_state = eef_state[:len(eef_state)//2]
                            arm2_eef_state = eef_state[len(eef_state)//2:]
                            arm1_gripper_state = [normalize_gripper_value(arm1_eef_state.pop())]
                            arm2_gripper_state = [normalize_gripper_value(arm2_eef_state.pop())]
                            if step_idx < steps-1:
                                eef_state_next = obsevations['arm_eef_state'][step_idx+1].tolist()
                                arm1_eef_state_next = eef_state_next[:len(eef_state)//2][0:-1]
                                arm2_eef_state_next = eef_state_next[len(eef_state)//2:][0:-1]

                                arm1_eef_action = [a-b for a,b in zip(arm1_eef_state_next,arm1_eef_state)]  #delta
                                arm2_eef_action = [a-b for a,b in zip(arm2_eef_state_next,arm2_eef_state)]
                            else:
                                arm1_eef_action = [0]*6  
                                arm2_eef_action = [0]*6  

                        if step_idx == steps-1:
                            terminal_flag = True
                        #打包observation
                        # pdb.set_trace()
                        step_data = Step(
                            observation=Observation(
                                # lang_instruction=task_name, 
                                arm1_joints_state=arm1_joints_state,
                                arm2_joints_state=arm2_joints_state,
                                arm1_gripper_state=arm1_gripper_state,
                                arm2_gripper_state=arm2_gripper_state,
                                arm1_eef_state=arm1_eef_state,
                                arm2_eef_state=arm2_eef_state,
                                base_state = base_state,
                                master_arm1_joints_state = master_arm1_joints_state,
                                master_arm2_joints_state = master_arm2_joints_state,
                                master_arm1_eef_state = master_arm1_eef_state,
                                master_arm2_eef_state = master_arm2_eef_state,
                                master_arm1_gripper_state=master_arm1_gripper_state,
                                master_arm2_gripper_state=master_arm2_gripper_state
                                ),
                            arm1_joints_action=arm1_joints_action,
                            arm2_joints_action=arm2_joints_action,
                            arm1_eef_action =arm1_eef_action,
                            arm2_eef_action =arm2_eef_action,
                            arm1_gripper_action=arm1_gripper_action,
                            arm2_gripper_action=arm2_gripper_action,
                            # base_action = base_action,
                            is_terminal = terminal_flag,

                            )
                        step_datas_list.append(step_data)
                    # pdb.set_trace()
                    episode_json = Episode(
                                    metadata=Metadata(
                                            dataset_name=dataset_name,
                                            episode_id=episode_id,
                                            scene=scene,
                                            environment=environment_name,
                                            task_name=task_name,
                                            num_steps=len(actions),
                                            robot_name="cobot_magic",
                                            robot_type="dual_arm",
                                            robot_description=robot_description,
                                            robot_arm1_joints_state_dim=episode_data['metadata']['robot_arm1_joints_state_dim'][0],
                                            robot_arm1_eef_state_dim=episode_data['metadata']['robot_arm1_eef_state_dim'][0],
                                            robot_arm1_eef_action_dim=episode_data['metadata']['robot_arm1_eef_action_dim'][0],
                                            robot_arm1_joints_action_dim=episode_data['metadata']['robot_arm1_joints_action_dim'][0],
                                            robot_arm1_gripper_state_dim=episode_data['metadata']['robot_arm1_gripper_state_dim'][0],
                                            robot_arm1_gripper_action_dim=episode_data['metadata']['robot_arm1_gripper_action_dim'][0],
                                            robot_arm2_joints_state_dim=episode_data['metadata']['robot_arm2_joints_state_dim'][0],
                                            robot_arm2_eef_state_dim=episode_data['metadata']['robot_arm2_eef_state_dim'][0],
                                            robot_arm2_eef_action_dim=episode_data['metadata']['robot_arm2_eef_action_dim'][0],
                                            robot_arm2_joints_action_dim=episode_data['metadata']['robot_arm2_joints_action_dim'][0],
                                            robot_arm2_gripper_state_dim=episode_data['metadata']['robot_arm2_gripper_state_dim'][0],
                                            robot_arm2_gripper_action_dim=episode_data['metadata']['robot_arm2_gripper_action_dim'][0],
                                            robot_master_arm1_eef_state_dim = episode_data['metadata']['robot_master_arm1_eef_state_dim'][0],
                                            robot_master_arm2_eef_state_dim = episode_data['metadata']['robot_master_arm2_eef_state_dim'][0],
                                            robot_master_arm1_joints_state_dim = episode_data['metadata']['robot_master_arm1_joints_state_dim'][0],
                                            robot_master_arm2_joints_state_dim = episode_data['metadata']['robot_master_arm2_joints_state_dim'][0],
                                            robot_master_arm1_gripper_state_dim = episode_data['metadata']['robot_master_arm1_gripper_state_dim'][0],
                                            robot_master_arm2_gripper_state_dim = episode_data['metadata']['robot_master_arm2_gripper_state_dim'][0],

                                            robot_base_action_dim=episode_data['metadata']['robot_base_action_dim'][0],
                                            robot_base_state_dim=episode_data['metadata']['robot_base_state_dim'][0],
                                            camera1_rgb_resolution = episode_data['metadata']['camera1_rgb_resolution'][:][0].tolist(),   #[h,w]
                                            camera2_rgb_resolution = episode_data['metadata']['camera2_rgb_resolution'][:][0].tolist(),
                                            camera3_rgb_resolution = episode_data['metadata']['camera3_rgb_resolution'][:][0].tolist(),
                                            camera5_rgb_resolution = episode_data['metadata']['camera4_rgb_resolution'][:][0].tolist(),
                                            # camera5_rgb_resolution = episode_data['metadata']['camera5_rgb_resolution'][:][0].tolist(),

                                            camera1_depth_resolution = episode_data['metadata']['camera1_depth_resolution'][:][0].tolist(),
                                            camera2_depth_resolution = episode_data['metadata']['camera2_depth_resolution'][:][0].tolist(),
                                            camera3_depth_resolution = episode_data['metadata']['camera3_depth_resolution'][:][0].tolist(),
                                            camera5_depth_resolution = episode_data['metadata']['camera4_depth_resolution'][:][0].tolist(),
                                            # camera5_depth_resolution = episode_data['metadata']['camera5_depth_resolution'][:][0].tolist(),
                                            experiment_time = exp_time,
                                            operator =operator,
                                            task_name_candidates = task_name_candidates,
                                            sample_rate = int(sample_rate)

                                        ),
                                    steps=step_datas_list
                                    # steps=[step_datas_list[0]]


                        )
                    # dataset_name = 'CobotMagic'
                    # pdb.set_trace()
                    scene_str = scene.replace(' ','')
                    # scene_str=scene
                    datafile_name = exp_time+'_'+dataset_name+'_'+robot_name+'_'+scene_str+'_'+environment_name+'_'+task_name+'_'+str(episode_id) 
                    # jsonstr = episode_json.json(indent=2)
                    # jsonstr = episode_json.json()
                    jsonstr = episode_json.dict()
                    # write json file and video file
                    # with open(f"{output_folder}/{datafile_name}.json", "w") as f:
                        # f.write(jsonstr)
                    # json_dict ={'q':"您好",
                    #             'vvv':'背景',
                    #             "ff":[1,2,3]}
                    with open(f"{output_folder}/{datafile_name}.json", "w",encoding='utf-8') as f:
                        json.dump(jsonstr, f,ensure_ascii=False,indent=2)
                    # pdb.set_trace()
                    save_video(cam_images_left, f"{output_folder}/{datafile_name}_camera1_rgb.mp4")
                    save_video(cam_images_right, f"{output_folder}/{datafile_name}_camera2_rgb.mp4")
                    # save_video(cam_images_high, f"{output_folder}//{datafile_name}_camera5_rgb.mp4")
                    save_video(cam_images_head, f"{output_folder}//{datafile_name}_camera5_rgb.mp4")
                    save_video(cam_images_front, f"{output_folder}//{datafile_name}_camera3_rgb.mp4")

                    save_depth_video(depthcam_images_left, f"{output_folder}/{datafile_name}_camera1_depth.mp4")
                    save_depth_video(depthcam_images_right, f"{output_folder}/{datafile_name}_camera2_depth.mp4")
                    # save_depth_video(depthcam_images_high, f"{output_folder}//{datafile_name}_camera5_depth.mp4")
                    save_depth_video(depthcam_images_head, f"{output_folder}//{datafile_name}_camera5_depth.mp4")
                    save_depth_video(depthcam_images_front, f"{output_folder}//{datafile_name}_camera3_depth.mp4")

                except IOError:
                    print("Error:读取文件失败！！")


if __name__ == '__main__':
    ainno_cobot_path = "/mnt/nas03/cobot_magic_datasets/data20240802"
    import pdb
    # pdb.set_trace()
    start_id = 1
    ainno_cobot_output_folder = "/mnt/nas03/cobot_magic_datasets/data20240802-ainno"
    mkdir_ifmissing(ainno_cobot_output_folder)
    covert_cobot_to_ainno(ainno_cobot_path,ainno_cobot_output_folder,start_id)
    