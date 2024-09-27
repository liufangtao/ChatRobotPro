import os
import json
import math
import cv2
import sys
import fire
import numpy as np
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_dir)

from ainnorobot_data.schema.episode_dataclass import Episode, Metadata, Observation, Step


class RealmanDataReader:

    def __init__(self, root_path):
        self.root_path = root_path
        self.target_size = (128, 128)  # w, h
        self.sample_rate = 30
        self.raw_sample_rate = 30

        self.robot_arm1_joints_state_dim = 6
        self.robot_arm1_eef_state_dim = 6
        self.robot_arm1_eef_action_dim = 6
        self.robot_lift_state_dim = 1
        self.robot_base_state_dim = 3
        self.robot_arm1_gripper_action_dim = 1
        self.robot_arm1_gripper_state_dim = 3
        self.robot_lift_action_dim = 1
        self.robot_base_action_dim = 2

        self.max_depth_value = 5100.0
        self.data_type_threshold_map = {
            'arm_and_lift_states': 200,
            'gripper_states': 500,
            'base_velocity_states': 500,
            'base_states': 500,
            'wrist_states': 200
        }

        self.dataset_name = 'realman'
        self.robot_name = 'Aixier'
        self.robot_type = 'single_arm'
        self.robot_description = 'arm1是机器人的机械臂, camera2是机械臂上的摄像头, camera5是机器人头部的摄像头, ' \
                                 'robot_arm1_joints_state_dim是关节状态, 从机械臂根部到夹爪一共有六个关节, 单位为弧度,' \
                                 'robot_arm1_eef_state_dim是机械臂位姿状态, 按顺序包含：x/y/z和rx/ry/rz x/y/z单位为米，rx/ry/rz单位为弧度,' \
                                 'robot_arm1_eef_action_dim是机械臂位姿控制状态, 按顺序包含：Δx/Δy/Δz/Δrx/Δry/Δrz Δx/Δy/Δz单位为米, /Δrx/Δry/Δrz 单位为弧度,' \
                                 'robot_lift_state_dim是升降机构状态 按顺序包含：升降机构位置, 单位为米,' \
                                 'robot_base_state_dim是机械臂位姿状态, 底盘的所有状态参数数量，按顺序包含：底盘x/y坐标值单位为米、底盘旋转角度单位为弧度,' \
                                 'robot_arm1_gripper_state_dim夹爪总的状态参数数量，按顺序包含：位移、速度、力,' \
                                 'robot_lift_action_dim是升降机构控制状态, 包含：升降机构位移（Δh） 单位为米'

    def read_video_time_stamp_data(self, file_name):
        """
        读取时间戳数据
        :param file_name: 时间戳文件名称
        :return:
        """
        time_stamp_list = []
        file_path = os.path.join(self.root_path, file_name)
        with open(file_path, 'r') as file:
            content = file.read()
        # 分割数据段
        data_segments = content.strip().split('\n')
        for segment in data_segments:
            time_stamp_index, time_stamp = segment.split(',')
            time_stamp_list.append({"timestamp": float(time_stamp), "frame_id": int(time_stamp_index)})
        return time_stamp_list

    def read_obs_data(self, file_name):
        """
        读取观测数据的txt文件
        :param file_name: 观测数据.txt文件, 可以是机械臂、夹爪、升降机构、底盘
        :return:
        """
        data_list = []
        file_path = os.path.join(self.root_path, file_name)
        # 遍历文件夹中的所有文件
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except Exception:
            raise ValueError(
                f"读取文件{file_path}时出现错误, 请检查文件")

        # 分割数据段
        data_segments = content.strip().split('\n\n')

        for segment in data_segments:
            data_dict = json.loads(segment)
            data_list.append(data_dict)
        return data_list

    def parse_log_file(self, file_path):
        result = {}
        file_path = os.path.join(self.root_path, file_path)
        with open(file_path, 'r') as file:
            lines = file.readlines()[1:-1]  # 跳过第一行和最后一行
        for line in lines:
            if 'task_name' in line:
                result['task_name'] = line.split('task_name:')[1].strip()
            if 'scene' in line:
                result['scene'] = line.split('scene:')[1].strip()
            if 'environment' in line:
                result['environment'] = line.split('environment:')[1].strip()
            if 'operator' in line:
                result['operator'] = line.split('operator:')[1].strip()
            if 'sample_rate' in line:
                result['sample_rate'] = int(line.split('sample_rate:')[1].strip())
        return result

    def get_lang_instruction(self, file_name='audio.txt'):
        """
        获取语音文件内容
        :return:
        """
        lang_instructions = []
        file_path = os.path.join(self.root_path, file_name)

        with open(file_path, 'r') as file:
            content = file.read()
        # 分割数据段
        data_segments = content.strip().split('\n')
        for segment in data_segments:
            time_stamp, lang_instruction = segment.split(',')
            if lang_instruction is not None:
                lang_instructions.append({
                    'timestamp': float(time_stamp),
                    'instruction': str(lang_instruction)
                })
        return lang_instructions

    def get_task_name_candidates(self, file_path=None):
        if not file_path:
            file_path = os.path.dirname( os.path.abspath(__file__))
        task_name_candidates_file_path = os.path.join(file_path, 'ainno_task_name_candidates.json')

        try:
            with open(task_name_candidates_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"File not found: {task_name_candidates_file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {task_name_candidates_file_path}")
            return None

    def find_left_nearest(self, timestamp_list, base_timestamp):
        idx = len(timestamp_list) - 1
        while timestamp_list[idx] > base_timestamp:
            idx -= 1
        return idx

    def find_right_nearest(self, timestamp_list, base_timestamp):
        idx = 0
        while idx < len(timestamp_list) and timestamp_list[idx] < base_timestamp:
            idx += 1
        found = False if idx == len(timestamp_list) else True
        return idx, found

    def align_all(self):
        obs_data = self.read_obs_data('robot_proprioception.txt')
        lang_instructions = self.get_lang_instruction('audio.txt')
        wrist_timestamps_data = self.read_video_time_stamp_data('timestamp_camera_wrist.txt')
        head_timestamps_data = self.read_video_time_stamp_data('timestamp_camera_head.txt')

        obs_data_timestamps = np.array([obs["timestamp"] for obs in obs_data])
        wrist_timestamps = np.array([d["timestamp"] for d in wrist_timestamps_data])
        head_timestamps = np.array([d["timestamp"] for d in head_timestamps_data])

        max_starts = max(obs_data_timestamps[0], wrist_timestamps[0])
        idx = 0
        while head_timestamps[idx] < max_starts:
            idx += 1
        start_timestamp = head_timestamps[idx]

        min_ends = min(obs_data_timestamps[-1], wrist_timestamps[-1])

        idx = len(head_timestamps) - 1
        while head_timestamps[idx] > min_ends:
            idx -= 1
        end_timestamp = head_timestamps[idx]

        # 获取重叠时间段内的帧
        head_mask = (head_timestamps >= start_timestamp) & (head_timestamps <= end_timestamp)
        head_timestamps_data_cut = np.array(head_timestamps_data)[head_mask]  # has be in order
        head_timestamps_cut = head_timestamps[head_mask]

        # align
        aligned_head_frame_id = [data["frame_id"] for data in head_timestamps_data_cut]
        aligned_wrist_frame_id = []
        aligned_obs_data = []

        for base_timestamp in head_timestamps_cut:
            # get aligned wrist frame id
            closest_target_index = np.argmin(np.abs(wrist_timestamps - base_timestamp))
            aligned_wrist_frame_id.append(wrist_timestamps_data[closest_target_index]["frame_id"])
            if abs(wrist_timestamps[closest_target_index] - base_timestamp) > self.data_type_threshold_map['wrist_states']:
                raise ValueError(
                    f"转化wrist_states数据: "
                    f"当前wrist_states时间戳:{wrist_timestamps[closest_target_index]}, "
                    f"当前图像时间戳:{base_timestamp}, "
                    f"差值为{abs(wrist_timestamps[closest_target_index] - base_timestamp):.4f} "
                    f"时间差值大于阈值{self.data_type_threshold_map['wrist_states']}."
                    f"数据废弃")

            # get aligned obs data
            index_left_nearest = self.find_left_nearest(obs_data_timestamps, base_timestamp)
            aligned_obs_data.append(obs_data[index_left_nearest])

            for key in self.data_type_threshold_map.keys():
                if key == 'wrist_states':
                    continue
                time_distance = abs(obs_data[index_left_nearest][key]['time'] - base_timestamp)
                if time_distance > self.data_type_threshold_map[key]:
                    raise ValueError(
                        f"转化{key}数据: "
                        f"当前{key}时间戳:{obs_data[index_left_nearest][key]['time']}, "
                        f"当前图像时间戳:{base_timestamp}, "
                        f"差值为{time_distance:.4f} "
                        f"时间差值大于阈值{self.data_type_threshold_map[key]}."
                        f"数据废弃")

        aligned_lang_instructions = [""] * len(head_timestamps_cut)
        for lang in lang_instructions:
            closest_target_index, found = self.find_right_nearest(head_timestamps_cut, lang['timestamp'])
            if found:
                aligned_lang_instructions[closest_target_index] = lang['instruction']

        assert len(aligned_head_frame_id) == len(aligned_wrist_frame_id) == len(aligned_obs_data) == len(aligned_lang_instructions)
        return aligned_head_frame_id, aligned_wrist_frame_id, aligned_obs_data, aligned_lang_instructions

    def calculate_eef_action(self, current_eef_states, next_eef_states):
        current_eef_states = np.asanyarray(current_eef_states)
        next_eef_states = np.asanyarray(next_eef_states)

        action_position = next_eef_states[:3] - current_eef_states[:3]
        # 处理正负180度处的突然符号切换, 将所有负值转换成正值
        f = lambda x: (x + 2 * math.pi) % (2 * math.pi)
        next_euler = f(next_eef_states[3:6])
        current_euler = f(current_eef_states[3:6])
        action_euler = next_euler - current_euler
        eef_action = np.concatenate([action_position, action_euler]).tolist()
        return eef_action

    def data_cook(self, aligned_obs_data):
        cooked_obs_data = []
        thresh_gripper_position_for_close = 5
        # 进行机械臂的action计算
        for index, data in enumerate(aligned_obs_data):
            action = dict()

            current_euler_eef = data['arm_and_lift_states']['euler']
            current_position_eef = data['arm_and_lift_states']['position']
            current_lift_state = data['arm_and_lift_states']['height'] / 1000.0
            current_eef_states = self.convert_to_meters(current_position_eef) + self.convert_to_theta(current_euler_eef)

            if index == len(aligned_obs_data) - 1:
                action['eef_action'] = [0, 0, 0, 0, 0, 0]
                action['lift_action'] = [0]
                action['base_action'] = [0, 0]
                action['gripper_action'] = [0]    # 0-open, 1-close
            else:
                next_data = aligned_obs_data[index + 1]
                next_eef_states = self.convert_to_meters(next_data['arm_and_lift_states']['position']) + \
                                  self.convert_to_theta(next_data['arm_and_lift_states']['euler'])
                next_lift_state = next_data['arm_and_lift_states']['height'] / 1000.0
                lift_action = np.array([next_lift_state]) - np.array([current_lift_state])

                action['eef_action'] = self.calculate_eef_action(current_eef_states, next_eef_states)
                action['lift_action'] = lift_action.tolist()
                action['base_action'] = [data["base_velocity_states"]['linear'], data["base_velocity_states"]['angular']]
                action['gripper_action'] = [1] if data['gripper_states']['position'] > thresh_gripper_position_for_close else [0]

            states = dict()
            states['base_state'] = [data['base_states']['x'], data['base_states']['y'], data['base_states']['theta']]
            states['lift_state'] = [current_lift_state]
            states['eef_state'] = current_eef_states

            base_position = self.map_range(data['gripper_states']['position'], 0, 255.0, 0, 0.075)
            base_velocity = self.map_range(data['gripper_states']['velocity'], 0, 255.0, 0, 0.1364)
            base_pressure = self.map_range(data['gripper_states']['pressure'], 0, 255.0, 40.0, 300.0)
            states['gripper_state'] = [base_position, base_velocity, base_pressure]

            states['joint_state'] = self.convert_to_theta(data['arm_and_lift_states']['joint_position'])

            cooked_obs_data.append({"states": states, "action": action})

        return cooked_obs_data

    def export_depth_video(self, depth_path_name, export_path, align_indices):
        """
        读取深度数据文件
        :param depth_path_name: 深度数据文件名称
        :param align_indices: 对齐后的时间戳
        :return:
        """
        abs_depth_path = self.root_path + f'/{depth_path_name}'
        if not os.path.exists(abs_depth_path):
            print(f"文件夹 {abs_depth_path} 不存在")
            return
        files = []
        # 遍历文件夹中的所有文件
        for file_index in align_indices:
            npy_file_path = os.path.join(abs_depth_path, f'{file_index}.npy')
            files.append(npy_file_path)
        video_writer = cv2.VideoWriter(f"{export_path}", cv2.VideoWriter_fourcc(*'mp4v'), self.sample_rate, self.target_size, isColor=False)
        try:
            for file_path in files:
                data = np.load(file_path)
                data = np.clip(data, a_min=0, a_max=self.max_depth_value).astype(np.float32) * 255.0 / self.max_depth_value
                frame = data.astype(np.uint8)
                frame = cv2.resize(frame, self.target_size)
                video_writer.write(frame)
        finally:
            video_writer.release()

    def convert_single_episode(self, episode_id, export_path):

        logger.info(f'start converter. episode_id:{episode_id}  export_path:{export_path}')

        task_name_candidates_map = self.get_task_name_candidates()

        # 对齐时间戳, 以头部摄像头为准
        aligned_head_frame_id, aligned_wrist_frame_id, aligned_obs_data, aligned_lang_instructions = self.align_all()
        obs_action_data = self.data_cook(aligned_obs_data)

        log_file_dict = self.parse_log_file('logs.txt')
        operator = log_file_dict.get('operator', 'unknown')
        scene = log_file_dict.get('scene', 'unknown')
        environment = log_file_dict.get('environment', 'unknown')
        task_name = log_file_dict.get('task_name', 'unknown')
        sample_rate = log_file_dict.get('sample_rate', self.raw_sample_rate)
        self.sample_rate = sample_rate
        dataset_name = self.dataset_name
        robot_name = self.robot_name
        experiment_time = os.path.basename(self.root_path)
        dataset_file_name = f"{experiment_time}_{dataset_name}_{robot_name}_{scene}_{environment}_{task_name}_{episode_id}"

        os.makedirs(export_path, exist_ok=True)
        with open(os.path.join(export_path, f"{dataset_file_name}.json"), "w") as f:
            episode = Episode(
                metadata=Metadata(
                    dataset_name=dataset_name,
                    episode_id=episode_id,
                    experiment_time=experiment_time,
                    operator=operator,
                    scene=scene,
                    environment=environment,
                    task_name=task_name,
                    task_name_candidates=task_name_candidates_map[task_name],
                    sample_rate=sample_rate,
                    num_steps=len(obs_action_data),
                    robot_name=robot_name,
                    robot_type=str(self.robot_type),
                    robot_description=str(self.robot_description),
                    robot_arm1_joints_state_dim=self.robot_arm1_joints_state_dim,
                    robot_arm1_eef_state_dim=self.robot_arm1_eef_state_dim,
                    robot_arm1_eef_action_dim=self.robot_arm1_eef_action_dim,
                    robot_lift_state_dim=self.robot_lift_state_dim,
                    robot_base_state_dim=self.robot_base_state_dim,
                    robot_arm1_gripper_action_dim=self.robot_arm1_gripper_action_dim,
                    robot_arm1_gripper_state_dim=self.robot_arm1_gripper_state_dim,
                    robot_lift_action_dim=self.robot_lift_action_dim,
                    robot_base_action_dim=self.robot_base_action_dim,
                    camera2_rgb_resolution=list(self.target_size)[::-1],
                    camera5_rgb_resolution=list(self.target_size)[::-1],
                    camera2_depth_resolution=list(self.target_size)[::-1],
                    camera5_depth_resolution=list(self.target_size)[::-1]
                ),
                steps=[
                    Step(
                        observation=Observation(
                            lang_instruction=aligned_lang_instructions[idx],
                            arm1_joints_state=obs_action['states']['joint_state'],
                            arm1_eef_state=obs_action['states']['eef_state'],
                            arm1_gripper_state=obs_action['states']['gripper_state'],
                            lift_state=obs_action['states']['lift_state'],
                            base_state=obs_action['states']['base_state'],
                        ),
                        arm1_eef_action=obs_action['action']['eef_action'],
                        arm1_gripper_action=obs_action['action']['gripper_action'],
                        lift_action=obs_action['action']['lift_action'],
                        base_action=obs_action["action"]['base_action'],
                        is_terminal=True if idx == len(obs_action_data) - 1 else False,
                        reward=1 if idx == len(obs_action_data) - 1 else 0,
                        discount=1
                    )
                    for idx, obs_action in enumerate(obs_action_data)
                ]
            )
            jsonstr = episode.json(indent=2, ensure_ascii=False)
            f.write(jsonstr)
        logger.info(f'===================================json文件生成成功=================================================')
        # 导出深度信息为mp4
        self.export_depth_video('camera_head_depth', f"{export_path}/{dataset_file_name}_camera2_depth.mp4", aligned_head_frame_id)
        self.export_depth_video('camera_wrist_depth', f"{export_path}/{dataset_file_name}_camera5_depth.mp4", aligned_wrist_frame_id)

        logger.info(f'===================================深度视频文件导出成功=================================================')
        head_video_writer = cv2.VideoWriter(f"{export_path}/{dataset_file_name}_camera2_rgb.mp4", cv2.VideoWriter_fourcc(*'avc1'), self.sample_rate, self.target_size)
        wrist_video_writer = cv2.VideoWriter(f"{export_path}/{dataset_file_name}_camera5_rgb.mp4", cv2.VideoWriter_fourcc(*'avc1'), self.sample_rate, self.target_size)

        head_cap = cv2.VideoCapture(os.path.join(self.root_path, 'camera_head_rgb.mp4'))
        wrist_cap = cv2.VideoCapture(os.path.join(self.root_path, 'camera_wrist_rgb.mp4'))

        head_frames = self.read_all_frames(head_cap)
        wrist_frames = self.read_all_frames(wrist_cap)

        try:
            for frame_id in aligned_head_frame_id:
                rgb_frame = cv2.resize(head_frames[frame_id], self.target_size)
                head_video_writer.write(rgb_frame)

            for frame_id in aligned_wrist_frame_id:
                rgb_frame = cv2.resize(wrist_frames[frame_id], self.target_size)
                wrist_video_writer.write(rgb_frame)
        finally:
            head_video_writer.release()
            wrist_video_writer.release()
        print(f'===================================rgb视频文件导出成功=================================================')

    def convert_to_meters(self, micron_list):
        """
        将列表中的所有微米值转换为米。

        参数:
        micron_list (list): 包含微米值的列表。

        返回:
        list: 包含米值的列表。
        """
        return [x / 1_000_000 for x in micron_list]

    def convert_to_theta(self, t):
        """
        欧拉角单位转换
        """
        return [x / 1_000 for x in t]

    def map_range(self, x, a, b, c, d):
        """
        将范围 [a, b] 中的数值 x 映射到范围 [c, d]
        :param x: 待转换的数值
        :param a: 原始范围的最小值
        :param b: 原始范围的最大值
        :param c: 目标范围的最小值
        :param d: 目标范围的最大值
        :return: 转换后的数值
        """
        return c + (x - a) * (d - c) / (b - a)

    def read_all_frames(self, cap):
        """
        读取摄像头的所有帧
        :param cap:
        :return:
        """
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames


def main(dataset_path, export_root_path):
    epsode_id = 0

    logger.add(
        f'{export_root_path}export.log',
        rotation='200 MB',
        level='INFO',
        format="{time:YYYY-MM-DD HH:mm:ss.SSS}|{message}",
        encoding='utf-8'
    )

    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            reader = RealmanDataReader(item_path)
            export_path = os.path.join(export_root_path, item)
            try:
                reader.convert_single_episode(epsode_id, export_path)
                epsode_id += 1
            except Exception as e:
                logger.error(e)
                continue


if __name__ == '__main__':
    fire.Fire(main)
