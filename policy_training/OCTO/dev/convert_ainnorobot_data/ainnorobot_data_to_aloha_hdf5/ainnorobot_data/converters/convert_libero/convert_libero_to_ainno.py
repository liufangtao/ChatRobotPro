import re
import cv2
import numpy as np
from ainnorobot_data.converters.convert_libero.libero import get_libero_dataset, get_libero_dataset_for_debugging
from ainnorobot_data.schema.episode_dataclass import Episode, Metadata, Observation, Step
from scipy.spatial.transform import Rotation as R
from datetime import datetime

def get_rotvec_orientation(quaternion_orientation):
    """
        Get axis angle representation from quaternion orientation.
        This function is copied from 
        https://gitlab.ainnovation.com/frontierresearch/ainnorobot/-/blob/main/baku/dataloader.py

        Copy code is UGLY, but this function is a math function. It should not change ever.
    """

    new_cartesian = []
    for i in range(len(quaternion_orientation)):
        pos = quaternion_orientation[i, :3]
        quat = quaternion_orientation[i, 3:7]
        ori = R.from_quat(quat).as_rotvec()
        if quaternion_orientation.shape[-1] > 7:
            new_cartesian.append(np.concatenate([pos, ori, quaternion_orientation[i, 7:]], axis=-1))
        else:
            new_cartesian.append(np.concatenate([pos, ori], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


def save_video(array, filename='output_video.mp4', fps=30):
    if len(array.shape) != 4 or array.shape[3] != 3:
        raise ValueError('Input array must have shape [frames, height, width, 3]')
    frames, height, width, channels = array.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(frames):
        frame = array[i]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        video_writer.write(frame)
    video_writer.release()


def covert_libero_to_ainno(dataset, ainno_libero_path):
    episode_id = 0
    for epsd_of_libero in dataset:
        episode_id += 1
        observations, actions, task_emb, file_path = epsd_of_libero
        assert len(actions) == len(observations['pixels'])

        pattern = r'/([^/]+)\.pkl$'
        match = re.search(pattern, file_path)
        if match:
            file_name = match.group(1)
        else:
            # TODO(zhangfaen): log warning instead of a simple print
            print("No file_name found.")
            continue
        
        scene = re.search(r'[_A-Z0-9]+', file_name).group(0).replace('_', ' ')
        task_name = re.search(r'[a-z][_a-z0-9]+$', file_name).group(0).replace('_', ' ')

        episode = Episode(
            metadata=Metadata(
                dataset_name="libero",
                episode_id=episode_id,
                experiment_time=datetime.now().strftime("%Y%m%d%H%M%S"),
                operator='zhangfaen',
                scene=scene,
                environment="libero",
                task_name=task_name,
                num_steps=len(actions),
                robot_name="libero",
                robot_type="single_arm",
                robot_description="libero",
                robot_arm1_eef_state_dim=6,
                robot_arm1_gripper_state_dim=2,
                robot_arm1_eef_action_dim=6,
                robot_arm1_gripper_action_dim=1,
            ),
            steps=[
                Step(
                    observation=Observation(
                        lang_instruction=task_name,
                        arm1_eef_state = get_rotvec_orientation(e.reshape(1, 7))[0].tolist(), # e.shape is (7),
                        arm1_gripper_state = g.tolist() # g.shape is (2, ),
                    ),
                    arm1_eef_action = ac.tolist()[:-1], # ac.shape is (7)
                    arm1_gripper_action = ac.tolist()[-1:], 
                ) for e, g, ac in zip(observations['eef_states'], observations['gripper_states'], actions)
            ]
        )
        jsonstr = episode.json(indent=2)
        experiment_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # write json file and video file
        with open(f"{ainno_libero_path}/{experiment_time_str}_libero_libero_{file_name}_{episode_id}.json", "w") as f:
            f.write(jsonstr)

        save_video(observations['pixels'], f"{ainno_libero_path}/{experiment_time_str}_libero_libero_{file_name}_{episode_id}_camera6_rgb.mp4")
        save_video(observations['pixels_egocentric'], f"{ainno_libero_path}/{experiment_time_str}_libero_libero_{file_name}_{episode_id}_camera2_rgb.mp4")


if __name__ == '__main__':
    # TODO(zhangfaen): replace below with get_libero_dataset
    dataset = get_libero_dataset("/mnt/nas03/robot_mixed_raw_data/libero")
    ainno_libero_path = "/mnt/nas03/AInnoRobotDatasets/third_party/libero"
    covert_libero_to_ainno(dataset=dataset, ainno_libero_path=ainno_libero_path)
    