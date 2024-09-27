import logging
import vla_server_pb2
import vla_server_pb2_grpc
import uuid
import time
import threading
from policy import make_policy


class BaseVLAServicer(vla_server_pb2_grpc.RobotServiceServicer):

    def __init__(self, config={}):
        self.config = config
        self.robot_type = vla_server_pb2.RobotType.Value(config.robot.robot_type)
        self.action_encoding = vla_server_pb2.ActionEncoding.Value(
            config.robot.action_encoding
        )
        self.state_encoding = vla_server_pb2.StateEncoding.Value(
            config.robot.state_encoding
        )
        self.connection_keep_alive = int(config.robot.connection_keep_alive)
        logging.info(f"robot_type = {config.robot.robot_type}")
        logging.info(f"action_encoding = {config.robot.action_encoding}")
        logging.info(f"state_encoding = {config.robot.state_encoding}")
        self.robots = {}
        self.lock = threading.Lock()
        self.maintenance_thread = threading.Thread(target=self._maintenance_worker)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()

        self.policy = make_policy(config.policy)

    def _is_robot_expired(self, robot_info):
        return (time.time() - robot_info["update_time"]) > robot_info["keep_alive"]

    def _maintenance_worker(self):
        while True:
            with self.lock:
                robot_ids_to_remove = []
                for robot_id, robot_info in self.robots.items():
                    if self._is_robot_expired(robot_info):
                        print(f"Removing expired robot with ID: {robot_id}")
                        robot_ids_to_remove.append(robot_id)

                for robot_id in robot_ids_to_remove:
                    del self.robots[robot_id]

            time.sleep(60)  # Check every minute

    def IsRobotAlive(self, robot_id):
        return robot_id in self.robots

    def UpdateState(self, robot_id):
        """将observation累积到历史状态中，并更新'update_time'"""
        if robot_id in self.robots:
            self.robots[robot_id]["update_time"] = time.time()

    def PolicySession(self, robot_id):
        if robot_id in self.robots:
            return self.robots[robot_id]["policy_session"]
        else:
            return None

    def CreateRobot(self, request, context):
        if request.robot_type != self.robot_type:
            logging.info(
                f"请求参数错误：支持的机器人类型为:{self.config.robot.robot_type}, 请求的机器人类型为:{vla_server_pb2.RobotType.Name(request.robot_type)}"
            )
            return vla_server_pb2.CreateRobotResponse(
                robot_id="",
                error_code=vla_server_pb2.ErrorType.ROBOT_TYPE_MISMATCH,
                message=f"支持的机器人类型为：{self.robot_type},请求的类型为：{request.robot_type}",
            )

        robot_id = str(uuid.uuid4())

        with self.lock:
            self.robots[robot_id] = {
                "description": request.description,
                "keep_alive": request.keep_alive or self.connection_keep_alive,
                "create_time": time.time(),
                "update_time": time.time(),
                "policy_session": self.policy.CreateSession(),  # 这个字段由具体内容由policy来维护
            }
            logging.info(
                f"Created robot with ID: {robot_id}, info={self.robots[robot_id]}"
            )

        return vla_server_pb2.CreateRobotResponse(robot_id=robot_id)
