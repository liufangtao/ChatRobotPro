from functools import partial
import logging
import os
import time

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 不要占用GPU

import argparse
import sys
import grpc
import numpy as np
import wandb
import vla_server_pb2
import vla_server_pb2_grpc
from PIL import Image
from io import BytesIO

import gym
from tqdm import trange


sys.path.append("../utils")


def rgb_array_to_jpeg_bytes(rgb_array, new_size=None):
    """
    将NumPy RGB数组转换为JPEG字节流。

    参数:
        rgb_array (numpy.ndarray): 一个形状为(height, width, 3)的NumPy数组，表示RGB图像。

    返回:
        bytes: JPEG格式的图像字节流。
    """
    # 确保输入是NumPy数组，并且有3个通道（RGB）
    if not isinstance(rgb_array, np.ndarray) or rgb_array.shape[2] != 3:
        raise ValueError("Input must be a NumPy array with shape (height, width, 3).")

    # 将NumPy数组转换为PIL图像
    image = Image.fromarray(rgb_array.astype(np.uint8), "RGB")
    if new_size is not None and rgb_array.shape[:2] != new_size[:2]:
        image = image.resize(new_size[:2][::-1])

    # 创建一个BytesIO流对象，并将PIL图像保存为JPEG格式到这个流中
    jpeg_bytes = BytesIO()
    image.save(jpeg_bytes, format="JPEG")

    # 获取JPEG字节流的值，并将流的位置重置到开始位置以便读取
    jpeg_data = jpeg_bytes.getvalue()
    jpeg_bytes.close()

    return jpeg_data


class RemotePolicy:
    def __init__(self, server_ip, server_port, robot_type=vla_server_pb2.ALOHA_HFLR):
        self._robot_type = robot_type
        channel = grpc.insecure_channel(f"{server_ip}:{server_port}")
        stub = vla_server_pb2_grpc.RobotServiceStub(channel)

        # 创建Robot实例的请求
        create_request = vla_server_pb2.CreateRobotRequest(
            # robot_type=vla_server_pb2.ALOHA_TLR,
            robot_type=self._robot_type,
            description="Example robot",
            keep_alive=60 * 10,
        )
        create_response = stub.CreateRobot(create_request)
        if create_response.error_code != vla_server_pb2.ErrorType.SUCCESS:
            raise Exception(
                f"创建机器人失败：error_code={create_response.error_code},message={create_response.message}"
            )
        else:
            print(f"Created robot with ID: {create_response.robot_id}")
            self._robot_id = create_response.robot_id
            self._stub = stub

    def _parse_obs(self, observation):
        if self._robot_type == vla_server_pb2.ALOHA_TLR:
            image_key_map = {
                "image_primary": "front",
                "image_left_wrist": "left_wrist",
                "image_right_wrist": "right_wrist",
            }
            proprio_key_map = {"proprio": "joints_state"}
        else:
            image_key_map = {
                "head": "head",
                "front": "front",
                "left_wrist": "left_wrist",
                "right_wrist": "right_wrist",
            }
            proprio_key_map = {"joints_state": "joints_state"}

        images = []
        proprioception = []

        for image_name, old_key in image_key_map.items():
            # 加载JPEG图像数据
            image_data = rgb_array_to_jpeg_bytes(
                np.array(observation["images"][old_key]), new_size=(480, 640, 3)
            )
            # 创建ImageData对象
            image_message = vla_server_pb2.ImageData(
                name=image_name, encoding="jpeg", data=image_data
            )
            images.append(image_message)

        for new_key, old_key in proprio_key_map.items():
            state = vla_server_pb2.ProprioceptionData(
                name=new_key,
                encoding=vla_server_pb2.StateEncoding.JOINT_BIMANUAL,
                data=observation["proprioception"][old_key],
            )
            proprioception.append(state)

        return {
            "images": images,
            "proprioception": proprioception,
            "time_step": observation.get("time_step", None),
        }

    def __call__(self, observation, language_instruction=None, unnorm_key=None):

        observation = self._parse_obs(observation)

        # 创建Observation对象
        observation_message = vla_server_pb2.Observation(
            images=observation["images"],
            proprioception=observation["proprioception"],
            time_step=observation.get("time_step", None),
        )

        task = vla_server_pb2.TaskInfo(language_instruction=language_instruction)

        # 创建RobotRequest对象
        robot_request = vla_server_pb2.RobotRequest(
            robot_id=self._robot_id, observation=observation_message, task=task
        )
        response = self._stub.ProcessRobotRequest(robot_request)
        if response.error_code == 0:

            return np.array([list(a.action) for a in response.result.actions])
        else:
            print("response={response}")


def run(
    robot_type=1,
    server_ip="127.0.0.1",
    server_port=58051,
    language_instruction=None,
    num_rollouts=10,
    max_steps=800,
    record=False,
    render=False,
    render_port=58056,
    env_name="aloha-sim-mix-v0",
):

    if env_name in [
        "aloha-sim-mix-v0",
        "aloha-sim-mix-delta-joints",
    ]:
        # keep this to register ALOHA sim env
        # 配置act路径
        sys.path.append("/nfsdata2/maohui/robotics/ALOHA/act-plus-plus")
        import envs.aloha_sim_env

    elif env_name in ["aloha-replay-v0"]:
        import envs.replay_env

        # env = gym.make("aloha-replay-v0")

    env = gym.make(env_name)

    if record:
        wandb.init(name="eval_aloha", project="octo", mode="offline")

    if render:
        from video_render.video_renderer import VideoRenderer

        renderer = VideoRenderer(port=render_port)
        renderer.start()

    # running rollouts
    for _ in range(num_rollouts):
        # 连接到gRPC服务器, 每个rollout重新连接，以创建新的会话。
        policy = RemotePolicy(server_ip, server_port, robot_type=robot_type)
        obs, info = env.reset()

        # run rollout for 400 steps
        # images = [obs["image_primary"][0]]
        def get_merged_image(obs):
            # return [obs["image_primary"][0]]
            # image_names = ["primary", "left_wrist","right_wrist","angle"]
            # image_names = ["image_"+x for x in image_names]
            # image_names = [x for x in image_names if x in obs]
            image_names = sorted(list(obs["images"].keys()))
            return np.concatenate([obs["images"][n] for n in image_names], axis=-2)

        images = [get_merged_image(obs)]
        episode_return = 0.0
        current_image = get_merged_image(obs)
        for _ in trange(max_steps):
            start_policy = time.time()
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy(obs, language_instruction)
            actions = actions[0]
            cost_policy = time.time() - start_policy

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            start_step = time.time()
            obs, reward, done, trunc, info = env.step(actions)
            cost_step = time.time() - start_step

            start_render = time.time()
            if render or record:
                current_image = get_merged_image(obs)
            if render:
                renderer.update_frame(current_image)
            if record:
                images.extend([current_image])
            cost_render = time.time() - start_render
            episode_return += reward
            if done or trunc:
                print(f"成功!")
                break

            cost = dict(
                policy=f"{cost_policy:.3f}s",
                step=f"{cost_step:.3f}s",
                render=f"{cost_render:.3f}s",
            )

            logging.debug(f"{cost}")
        print(f"Episode return: {episode_return}")
        if record:
            # log rollout video to wandb -- subsample temporally 2x for faster logging
            images = np.array(images)
            wandb.log(
                {
                    "rollout_video": wandb.Video(
                        images.transpose(0, 3, 1, 2)[::2], fps=60, format="mp4"
                    )
                }
            )


def select_task():

    selection_map = {
        "1": "Use the left arm to hold the left cup (which contains one red ball) and the right arm to hold the right cup. Pour the red ball from the left cup into the right cup. Finally, place both cups in the middle of the table.",
        "2": "暂无",
    }
    selection = "\n".join([f"{k:2} :  {v}" for k, v in selection_map.items()])
    while True:
        s = input(selection + "\n\n请选择：")
        if s in selection_map:
            print(f"您的选择是: {s}: {selection_map[s]}\n")
            break
        else:
            print("请在以下选项中选择！")
    return selection_map[s]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gRPC client for RobotService")
    parser.add_argument(
        "--robot_type", type=int, default=2, choices=[1, 2], help="机器人类型"
    )
    parser.add_argument(
        "--ip", type=str, default="127.0.0.1", help="IP address of the gRPC server"
    )
    parser.add_argument(
        "--port", type=int, default=50051, help="Port number of the gRPC server"
    )
    parser.add_argument("--max_steps", type=int, default=250, help="最大步数")
    parser.add_argument("--num_rollouts", type=int, default=1, help="num_rollouts")
    parser.add_argument("--render", type=int, default=0, help="是否显示视频")
    parser.add_argument("--record", type=int, default=0, help="是否记录视频")
    parser.add_argument("--render_port", type=int, default=50056, help="显示视频的端口")
    parser.add_argument(
        "--env_name",
        type=str,
        default="aloha-replay-v0",
        choices=[
            "aloha-sim-mix-v0",
            "aloha-replay-v0",
            "aloha-sim-mix-delta-joints",
        ],
        help="环境",
    )

    parser.add_argument("--task", type=str, default=None, help="任务指令")
    parser.add_argument(
        "--logging", type=str, default="cfgs/logging.yaml", help="logging config file"
    )
    args = parser.parse_args()
    print(f"args={args}")


    # 配置日志
    with open(args.logging, "r") as f:
        import logging.config
        import yaml

        config = yaml.safe_load(f)
        logging.config.dictConfig(config)


    run = partial(
        run,
        robot_type=args.robot_type,
        server_ip=args.ip,
        server_port=args.port,
        num_rollouts=args.num_rollouts,
        record=args.record,
        render=args.render,
        render_port=args.render_port,
        max_steps=args.max_steps,
        env_name=args.env_name,
    )

    if args.task is None:
        args.task = select_task()

    run(language_instruction=args.task)
    # run(task='transfer_coffe')
    # run(task='transfer_cube')
    # run(task='insertion')
    # run(task='transfer_peg')
