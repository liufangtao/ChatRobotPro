import logging
import os

import argparse
from collections import defaultdict
import sys
import time
from typing import Any, Dict, Optional, Union
from PIL import Image
import vla_server_pb2
import vla_server_pb2_grpc
import numpy as np
from PIL import Image
from io import BytesIO

from vla_server_pb2 import ErrorType, RobotType

import grpc
from concurrent import futures
from base_vla_server import BaseVLAServicer


sys.path.append("../utils")
from video_render.video_renderer import VideoRenderer


class VLAServicer(BaseVLAServicer):
    def __init__(self, config):
        self._config = config
        super().__init__(config)
        self._enable_renderer = config.get("renderer", {}).get("enabled", False)

        if self._enable_renderer:
            self._renderer = VideoRenderer(
                host=config.renderer.ip, port=config.renderer.port
            )
            self._renderer.start()

    def _convert_robot_request(self, robot_request):
        """将RobotRequest对象转换为Python字典，并将图像数据解码为NumPy数组"""

        def decode_image_data(image_data, encoding="jpeg"):
            """将image_data中的二进制数据解码为NumPy数组"""
            if encoding == "jpeg":
                image_stream = BytesIO(image_data)
                image = Image.open(image_stream)
                return np.array(image)
            else:
                raise NotImplementedError

        converted_request_dict = {}

        converted_request_dict["language_instruction"] = (
            robot_request.task.language_instruction
        )

        # 遍历所有图像数据，将二进制数据解码为NumPy数组
        images = dict()
        for image_data in robot_request.observation.images:
            image_np = decode_image_data(image_data.data, image_data.encoding)
            image_name = image_data.name
            images[image_name] = image_np

        # 遍历所有proprioception数据，将二进制数据解码为NumPy数组
        states = dict()
        for proprio_data in robot_request.observation.proprioception:
            if proprio_data.encoding == self.state_encoding:
                _name = proprio_data.name
                _data = np.array(proprio_data.data)
                # proprioceptions[_name]=_data
                states[_name] = _data
                # converted_request_dict["observation"]["proprio"]=_data
                break

        converted_request_dict["observation"] = {
            "images": images,
            "states": states,
            # "time_step":robot_request.observation.time_step,
        }
        if hasattr(robot_request.observation, "time_step"):
            converted_request_dict["observation"][
                "time_step"
            ] = robot_request.observation.time_step

        return converted_request_dict

    def ProcessRobotRequest(self, request, context):

        # 构建响应消息
        response = vla_server_pb2.RobotResponse()

        if self.IsRobotAlive(request.robot_id):
            start = time.time()

            self.UpdateState(request.robot_id)

            start_decoding = time.time()
            converted_request = self._convert_robot_request(request)
            obs = converted_request["observation"]
            instruction = converted_request["language_instruction"]
            unnorm_key = converted_request.get("unnorm_key", None)
            cost_decoding = time.time() - start_decoding

            if self._enable_renderer:
                start_renderer = time.time()
                merged_image = np.concatenate(
                    [obs["images"][k] for k in sorted(obs["images"])], axis=-2
                )
                self._renderer.update_frame(merged_image)
                cost_renderer = time.time() - start_renderer

            time_policy_start = time.time()
            actions = self.policy(
                session=self.PolicySession(request.robot_id),
                observation=obs,
                language_instruction=instruction,
                unnorm_key=unnorm_key,
            )
            time_policy_cost = time.time() - time_policy_start

            start_preparing_results = time.time()
            response.error_code = ErrorType.SUCCESS  # 表示成功
            response.message = "success"
            response.result.actions.extend(
                [vla_server_pb2.ActionData(action=r) for r in actions.tolist()]
            )  # 将动作列表添加到响应中
            response.result.encoding = self.action_encoding
            cost_preparing_results = time.time() - start_preparing_results

            start_log_info = time.time()
            proprio = obs["states"]
            info = dict(
                # policy={type(self.policy)},
                instruction=instruction,
                obs=set(obs.keys()),
                proprio=str(proprio).replace("\n", " "),
                action_info=(type(actions), actions.shape),
                actions=str(actions[0]).replace("\n", " "),
            )
            logging.debug(f"{info}")
            cost_log_info = time.time() - start_log_info

            cost = time.time() - start
            cost_info = dict(
                total=f"{cost:.3f}",
                decoding=f"{cost_decoding:.3f}",
                policy=f"{time_policy_cost:.3f}",
                response=f"{cost_preparing_results:.3f}",
                log=f"{cost_log_info:.3f}",
            )
            if self._enable_renderer:
                cost_info["renderer"] = f"{cost_renderer:.3f}"

            logging.info(
                f"cost={cost_info},robot={request.robot_id},instruction={instruction}"
            )
        else:
            response.error_code = ErrorType.ROBOT_NOT_EXIST  # 不存在
            response.message = f"机器人{request.robot_id}不存在，可能是由于长时间未连接导致机器人状态过期。"
            # response.result.action.extend(actions)  # 将动作列表添加到响应中
            logging.warn(f"ROBOT_NOT_EXIST:{request.robot_id}")

        return response


def main(cfg):
    ip = cfg.service.ip
    port = cfg.service.port
    workers = cfg.service.workers
    # log_file=cfg.service.log
    print(f"config={cfg}")

    logging.info(f"Server starting on {ip}:{port}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    vla_server_pb2_grpc.add_RobotServiceServicer_to_server(VLAServicer(cfg), server)
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    logging.info(f"Server started on {ip}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gRPC server for RobotService")
    parser.add_argument(
        "-c", "--config", type=str, default="cfgs/config.yaml", help="config file"
    )
    parser.add_argument(
        "--logging", type=str, default="cfgs/logging.yaml", help="logging config file"
    )
    parser.add_argument("--gpu", type=str, default=None, help="指定GPU卡")
    args = parser.parse_args()
    print(f"cwd={os.getcwd()}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)


    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.50' #限制XLA占用显存

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 配置日志
    with open(args.logging, "r") as f:
        import logging.config
        import yaml

        config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    main(cfg)
