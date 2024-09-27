import os
import h5py
import pickle as pkl
import numpy as np
from pathlib import Path
import cv2
import time
from concurrent.futures import ProcessPoolExecutor
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
# load sentence transformer
print("loading sentence transformer")
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("loaded.")
 

def process_task_ee_env(task_file):
    benchmark, task_file, ds_factor = task_file
    # print(f"Processing {task_file} for {benchmark}")
    with h5py.File(task_file, 'r') as root:

        observation = {}
        observation["qpos"] = np.array(root['/observations/ee_qpos_states'][:], dtype=np.float32)
        observation["cartesian_states"] = np.array(root['/observations/ee_cartesian_states'][:], dtype=np.float32)
        observation["gripper_states"] = np.array(root['/observations/ee_gripper_states'][:], dtype=np.float32)

        image_dict = dict()
        for cam_name in camera_names:
            imgs = root[f'/observations/ee_images/{cam_name}'][:]
            new_width = imgs.shape[2] // ds_factor
            new_height = imgs.shape[1] // ds_factor
            images = np.array([cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA).astype(np.uint8) for img in imgs]) if ds_factor!=1 else imgs
            observation[cam_name] = np.array(images, dtype=np.uint8)

        return {
            "observations": [observation],
            "actions": [np.array(root['/ee_actions'][:], dtype=np.float32)],
        }


def process_task_joints_env(task_file):
    benchmark, task_file, ds_factor = task_file
    # print(f"Processing {task_file} for {benchmark}")
    with h5py.File(task_file, 'r') as root:
        
        # camera_names = ['top', 'left_wrist', 'right_wrist']

        observation = {}
        observation["qpos"] = np.array(root['/observations/qpos'][:], dtype=np.float32)
        # image_dict = dict()
        for cam_name in camera_names:
            imgs = root[f'/observations/images/{cam_name}']
            new_width = imgs.shape[2] // ds_factor
            new_height = imgs.shape[1] // ds_factor
            images = np.array([cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA).astype(np.uint8) for img in imgs]) if ds_factor!=1 else imgs
            observation[cam_name] = np.array(images, dtype=np.uint8)

        return {
            "observations": [observation],
            "actions": [np.array(root['/action'][:], dtype=np.float32)],
        }

def get_data(benchmark,ds_factor=4,env_is_joints=True):
    print(f"############################# {benchmark} #############################")
    benchmark_path = DATASET_PATH / benchmark
    task_files = list(benchmark_path.glob("*.hdf5"))
    print("task_files_len=",len(task_files))

    t1 = time.time()
    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        process_task  = process_task_joints_env if env_is_joints else process_task_ee_env
        results = list(executor.map(process_task, [(benchmark, str(task_file), ds_factor) for task_file in task_files]))
        gc.collect()
        
    print(f"Processed {len(task_files)} files in {time.time() - t1:.2f} seconds")
    # Combine the results from all processes
    observations = []
    actions = []
    for result in results:
        observations.extend(result["observations"])
        actions.extend(result["actions"])

    data = {
            "observations": observations,
            "states": [],
            "actions": actions,
            "task_emb": lang_model.encode(language_instruction_dict[benchmark]),
        }

    return data



if __name__ == "__main__":

    # language = "en"
    language = "cn"

    ds_factor= 1
    # env_is_joints=True
    env_is_joints=False

    camera_names = ['top', 'left_wrist', 'right_wrist', 'angle']

    ############ 2arm
    DATASET_PATH = Path(f"baku_aloha_2arm_ep200")
    ############ 1arm

    BENCHMARKS = ["transfer_cube", "insertion"]
    # BENCHMARKS = ["transfer_cube"]
    # BENCHMARKS = ["insertion"]

    ##########+++++++++++++++++++++++++++++++++++++++++++++++++

    language_instruction_dict = {
                "transfer_cube": "将绿色方块夹起来然后交给另一个机械臂",
                "insertion": "将红色插销插入到蓝色插槽中"
            } if language == "cn" else {
                "transfer_cube": "transfer the green cube",
                "insertion": "insert peg into socket"
            }

    mode ="joints" if env_is_joints else "ee"
    h=str(480//ds_factor)
    w=str(640//ds_factor)

    cleaned_names = [name.replace('_wrist', '') for name in camera_names]
    camera_string = '-'.join(cleaned_names)
    print(camera_string)
    ##########---------------------------------------------------
    SAVE_DATA_PATH = Path(f"./expert_demos/aloha_h{h}w{w}_{mode}_{camera_string}_{DATASET_PATH.name}_{language}")

    # create save directory
    SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"ds_factor: {ds_factor}, env_is_joints: {env_is_joints}, SAVE_DATA_PATH:{str(SAVE_DATA_PATH)}")
    for benchmark in BENCHMARKS:
        data  = get_data(benchmark,ds_factor,env_is_joints)
        gc.collect()
        save_data_path = SAVE_DATA_PATH / (benchmark + ".pkl")
        with open(save_data_path, "wb") as f:
            pkl.dump(data, f)
        print(f"Saved to {str(save_data_path)}")
