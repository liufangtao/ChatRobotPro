#--coding:utf-8--
import json
import copy
import os.path

import cv2
import time
import socket
import threading
import multiprocessing
import pyrealsense2 as rs
import numpy as np

from ainnorobot_data.collector_realman.robotic_arm_package.robotic_arm import *


class GripperCollectThread(threading.Thread):

    def __init__(self, client, stop_event, logger=None):
        super().__init__()
        self.client = client
        self.logger = logger
        self.newest_states = None
        self.stop_event = stop_event

    def run(self):
        try:
            if self.logger is not None:
                self.logger.info("Gripper collector started!")
            states = None
            while not self.stop_event.is_set():
                response = self.client.recv(10240).decode().strip("\n")
                if len(response) > 0:
                    results = response.split("\n")
                    for res in results:
                        try:
                            data = json.loads(res)
                        except Exception as e:
                            self.logger.info(f"arm and lift state data is damaged. Exception: {e}")
                            continue
                        if "command" in data and data['command'] == 'read_multiple_holding_registers':
                            if 'data' in data and len(data['data']) >= 8:
                                states = {'position': data['data'][2],
                                          'pressure': data['data'][4],
                                          'velocity': data['data'][5],
                                          'time': time.time() * 1000}
                            self.newest_states = copy.deepcopy(states)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f'夹爪状态数据接收出错，退出夹爪状态采集！Exception: {e}')

    def get_observation(self):
        return self.newest_states


class GripperSendThread(threading.Thread):

    def __init__(self, client, sample_frequency, stop_event, logger=None):
        super().__init__()
        self.client = client
        self.sample_frequency = sample_frequency
        self.logger = logger
        self.stop_event = stop_event

    def run(self):
        step_time = 1.0 / self.sample_frequency
        sleep_time = 0.00001

        try:
            if self.logger is not None:
                self.logger.info("Gripper sender started!")
            while not self.stop_event.is_set():
                time.sleep(sleep_time)
                t1 = time.time()
                data = '{"command":"read_multiple_holding_registers","port":1,"address":2000,"num":6,"device":9}\r\n'
                ret = self.client.send(data.encode("utf-8"))
                t2 = time.time()
                sleep_time = max(0.00001, step_time - (t2 - t1))
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f'读取夹爪数据指令发送出错，退出夹爪状态采集！Exception: {e}')


class Gripper(object):
    def __init__(self, gripper_ip, gripper_port, sample_frequency=30, logger=None):
        super().__init__()
        self.sample_frequency = sample_frequency
        self.logger = logger

        # init communication
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((gripper_ip, gripper_port))
        data = '{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout ":2}\r\n'
        self.client.send(data.encode("utf-8"))

        self.collector_thread_stop_event = threading.Event()
        self.collector = GripperCollectThread(self.client, self.collector_thread_stop_event, logger)
        self.sender_thread_stop_event = threading.Event()
        self.sender = GripperSendThread(self.client, self.sample_frequency, self.sender_thread_stop_event, logger)
        self.stopped = False

    def control(self, displacement=None):
        if displacement is not None:
            data = '{"command":"write_registers","port":1,"address":1000,"num":3,"data":' + f"[0, 9, {displacement}, 0, 128, 255]," + '"device":9}\r\n'
            self.client.send(data.encode("utf-8"))
        else:
            pass

    def start(self):
            self.collector.start()
            self.sender.start()
            self.logger.info("Gripper started!")

    def stop(self):
        if not self.stopped:
            self.sender_thread_stop_event.set()
            self.collector_thread_stop_event.set()
            self.collector.join()
            self.logger.info("Gripper collector stopped!")
            self.sender.join()
            self.logger.info("Gripper sender stopped!")
            self.client.close()
            self.logger.info("Gripper stopped!")
            self.stopped = True

    def stop_read(self):
        if not self.stopped:
            self.stop()

    def stop_write(self):
        if not self.stopped:
            self.stop()

    def get_observation(self):
        return self.collector.get_observation()


class ArmCollectThread(threading.Thread):
    def __init__(self, ip, port, stop_event, logger=None):
        super().__init__()
        self.ip = ip
        self.port = port
        self.stop_event = stop_event
        self.logger = logger
        self.newest_states = None

    def run(self):
        m_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            m_socket.bind((self.ip, self.port))
            if self.logger is not None:
                self.logger.info("ArmAndLift collector started!")
            while not self.stop_event.is_set():
                info, _ = m_socket.recvfrom(1024)
                try:
                    data = json.loads(info)
                except Exception as e:
                    self.logger.info(f"arm and lift state data is damaged. Exception: {e}")
                    continue
                if 'waypoint' in data and 'joint_status' in data and 'lift_state' in data:
                    states = {'position': data['waypoint']['position'],
                              'euler': data['waypoint']['euler'],
                              'quat': data['waypoint']['quat'],
                              'joint_position': data['joint_status']['joint_position'],
                              'joint_en_flag': data['joint_status']['joint_en_flag'],
                              'joint_err_code': data['joint_status']['joint_err_code'],
                              'height': data['lift_state']['height'],
                              'time': time.time() * 1000}
                    print("arm receive time: ", states["time"])
                    self.newest_states = copy.deepcopy(states)
        except KeyboardInterrupt:
            self.logger.warn("KeyboardInterrupt: ArmAndLift collector")
        except Exception as e:
            self.logger.error(f"Abnormal exit: ArmAndLift collector. Exception: {e}")
        finally:
            m_socket.close()

    def get_observation(self):
        return self.newest_states


class ArmAndLift(object):
    def __init__(self, controller_ip, host_ip, host_port, logger=None):
        super().__init__()
        self.logger = logger
        self.operator = Arm(RM65, controller_ip)
        ret = -1
        while ret != 0:
            ret = self.operator.Set_Realtime_Push(cycle=5, port=host_port, enable=True, ip=host_ip)

        self.thread_stop_event = threading.Event()
        self.capture_thread = ArmCollectThread(host_ip, host_port, self.thread_stop_event, logger)

    def control(self, height=None, pose=None):
        status = True
        if height is not None:
            ret = self.operator.Set_Lift_Height(height, speed=30)
            status = (status and False) if ret != 0 else status
        if pose is not None:
            ret = self.operator.Movep_CANFD(pose, False)
            status = (status and False) if ret != 0 else status
        return status

    def control_lift(self, speed=0):
        self.operator.Set_Lift_Speed(speed)

    def start(self):
        self.capture_thread.start()
        if self.logger is not None:
            self.logger.info("ArmAndLift started!")

    def stop(self):
        self.thread_stop_event.set()
        self.capture_thread.join()
        self.logger.info("ArmAndLift collector stopped!")
        self.logger.info("ArmAndLift stopped!")

    def stop_read(self):
        self.thread_stop_event.set()
        self.capture_thread.join()
        self.logger.info("ArmAndLift collector stopped!")
        self.logger.info("ArmAndLift read stopped!")

    def stop_write(self):
        self.logger.info("ArmAndLift write stopped!")

    def get_observation(self):
        return self.capture_thread.get_observation()

    def init_pose(self, pose=[0.329, -0.0124, 0.248, -2.783, 0.059, -1.761], height=435):
        self.operator.Movel_Cmd(pose, 30, 0, 0, True)
        self.operator.Set_Lift_Height(height, 50, True)


class BaseCollectThread(threading.Thread):
    def __init__(self, ip, port, sample_frequency, stop_event, logger=None):
        super().__init__()
        self.ip = ip
        self.port = port
        self.sample_frequency = sample_frequency
        self.logger = logger

        self.client = None

        self.newest_states = None
        self.newest_velocity = None
        self.stop_event = stop_event

    def run(self):
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.ip, self.port))
            self.client.send(f'/api/request_data?topic=robot_status&switch=on&frequency={self.sample_frequency}'.encode('utf-8'))
            self.client.send(f'/api/request_data?topic=robot_velocity&switch=on&frequency={self.sample_frequency}'.encode('utf-8'))
            if self.logger is not None:
                self.logger.info("Base collector started!")
            while not self.stop_event.is_set():
                response = self.client.recv(10240).decode().strip("\n")
                if len(response) > 0:
                    results = response.split("\n")
                    for res in results:
                        try:
                            data = json.loads(res)
                        except Exception as e:
                            self.logger.info(f"base state data is damaged. Exception: {e}")
                            continue
                        if data['type'] == 'callback' and data["topic"] == "robot_status":
                            states = {'x': data['results']['current_pose']['x'],
                                      'y': data['results']['current_pose']['y'],
                                      'theta': data['results']['current_pose']['theta'],
                                      'time': time.time() * 1000}
                            self.newest_states = copy.deepcopy(states)
                        if data['type'] == 'callback' and data["topic"] == "robot_velocity":
                            states = {'angular': data['results']['angular'],
                                      'linear': data['results']['linear'],
                                      'time': time.time() * 1000}
                            self.newest_velocity = copy.deepcopy(states)
                else:
                    raise ValueError("Base collector 接收数据为空")
        except KeyboardInterrupt:
            self.logger.warn("KeyboardInterrupt: base collector")
        except Exception as e:
            self.logger.error(f'Abnormal exit: base collector. Exception:{e}')
        finally:
            self.client.close()

    def send(self, msg):
        self.client.send(msg.encode('utf-8'))

    def get_observation(self):
        return self.newest_states, self.newest_velocity


class Base(object):
    def __init__(self, ip, port, sample_frequency=2, logger=None):
        self.ip = ip
        self.port = port
        self.sample_frequency = sample_frequency
        self.logger = logger

        self.thread_stop_event = threading.Event()
        self.collector = BaseCollectThread(ip, port, sample_frequency, self.thread_stop_event, logger)

    def start(self):
        self.collector.start()
        self.logger.info("Base started!")

    def stop(self):
        self.thread_stop_event.set()
        self.collector.join()
        self.logger.info("Base collector stopped!")
        self.logger.info("Base stopped!")

    def stop_read(self):
        self.thread_stop_event.set()
        self.collector.join()
        self.logger.info("Base collector stopped!")
        self.logger.info("Base read stopped!")

    def stop_write(self):
        self.logger.info("Base write stopped!")

    def control(self, linear_velocity, angular_velocity):
        msg = f'/api/joy_control?angular_velocity={angular_velocity}&linear_velocity={linear_velocity}'
        self.collector.send(msg)

    def get_observation(self):
        return self.collector.get_observation()


def save_depth(depth_queue, save_dir, time_filename, save_data, stop_event, logger=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    count = 0
    try:
        if logger is not None:
            logger.info("Depth saver started!")
        with open(time_filename, "a") as wf:
            while not stop_event.is_set():
                if depth_queue.empty():
                    time.sleep(0.01)
                    continue
                depth_frame = depth_queue.get()
                if depth_frame is None:
                    break
                if save_data:
                    timestamp = depth_frame["time"]
                    wf.write(f'{count},{timestamp}\n')
                    depth_frame = depth_frame["depth"]
                    if depth_frame is not None:
                        save_filepath = os.path.join(save_dir, f"{count}.npy")
                        np.save(save_filepath, depth_frame)
                        count += 1
    except KeyboardInterrupt:
        logger.warn("KeyboardInterrupt: Depth saver.")
    except Exception as e:
        logger.error(f'Abnormal exit: Depth saver. Exception:{e}')
    finally:
        logger.info(f'深度数据保存总帧数：{count}')


def save_color(color_queue, save_dir, name, fps, width, height, save_data, stop_event, logger=None):
    count = 0
    filename = os.path.join(save_dir, f"camera_{name}_rgb.mp4")
    vw = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    try:
        if logger is not None:
            logger.info("RGB saver started!")
        while not stop_event.is_set():
            if color_queue.empty():
                time.sleep(0.01)
                continue
            color_frame = color_queue.get()
            if color_frame is None:
                break
            if save_data and color_frame is not None:
                vw.write(color_frame)
                count += 1
    except KeyboardInterrupt:
        logger.warn("KeyboardInterrupt: RGB saver.")
    except Exception as e:
        logger.error(f'Abnormal exit: RGB saver. Exception:{e}')
    finally:
        logger.info(f'RGB数据保存总帧数：{count}')
        vw.release()


class VideoCollectThread(threading.Thread):
    """
    深度摄像头视频帧读取
    """

    def __init__(self, pipeline, depth_queue, color_queue, stop_event, logger=None, show=1):
        super().__init__()
        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color)
        self.depth_queue = depth_queue
        self.color_queue = color_queue
        self.transfer = False
        self.logger = logger
        self.show = show
        self.newest_rgb = None
        self.newest_depth = None
        self.stop_event = stop_event

    def run(self):
        count = 0
        try:
            if self.logger is not None:
                self.logger.info("Video collector started!")
            window_name = 'RGB'
            while not self.stop_event.is_set():
                count += 1
                # print(f'采集第{count}帧')
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                depth = np.asanyarray(depth_frame.get_data())
                rgb = np.asanyarray(color_frame.get_data())
                self.depth_queue.put({"depth": depth, "time": color_frame.get_timestamp()})
                self.color_queue.put(rgb)
                # self.time_queue.put(color_frame.get_timestamp())
                self.newest_depth = copy.deepcopy(depth)
                self.newest_rgb = copy.deepcopy(rgb)

                if self.show == 0:
                    cv2.imshow(window_name, np.asanyarray(color_frame.get_data()))
                    cv2.waitKey(1)
        except KeyboardInterrupt:
            self.logger.warn("KeyboardInterrupt: video collector")
        except Exception as e:
            self.logger.error(f'Abnormal exit: video collector. Exception:{e}')
        finally:
            self.logger.info(f'采集总帧数:{count}')
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def get_observation(self):
        return self.newest_rgb, self.newest_depth


class Camera:
    def __init__(self, save_dir, name, sample_rate, width, height, save_data=True, show=1, logger=None):
        self.save_dir = save_dir
        self.name = name
        self.fps = sample_rate
        self.width = width
        self.height = height
        self.show = show
        self.logger = logger

        self.pipeline = self.init_pipeline()
        self.depth_queue = multiprocessing.Queue()
        self.color_queue = multiprocessing.Queue()
        self.depth_process_stop_event = multiprocessing.Event()
        self.depth_process = multiprocessing.Process(target=save_depth, args=(self.depth_queue, os.path.join(save_dir, f"camera_{name}_depth"), os.path.join(save_dir, f'timestamp_camera_{name}.txt'), save_data, self.depth_process_stop_event, logger))
        self.depth_process.daemon = True
        self.color_process_stop_event = multiprocessing.Event()
        self.color_process = multiprocessing.Process(target=save_color, args=(self.color_queue, save_dir, name, self.fps, width, height, save_data, self.color_process_stop_event, logger))
        self.color_process.daemon = True
        self.collector_thread_stop_event = threading.Event()
        self.collector = VideoCollectThread(self.pipeline, self.depth_queue, self.color_queue, self.collector_thread_stop_event, logger, show)

    def init_pipeline(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        pipeline = rs.pipeline()
        config = rs.config()
        if self.name == 'head':
            serial = devices[0].get_info(rs.camera_info.serial_number)
            self.logger.info(f"head serial: {serial}")
        else:
            serial = devices[1].get_info(rs.camera_info.serial_number)
            self.logger.info(f"wrist serial: {serial}")

        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipeline.start(config)
        return pipeline

    def start(self):
        self.collector.start()
        self.depth_process.start()
        self.color_process.start()
        self.logger.info("Camera started!")

    def stop(self):
        self.collector_thread_stop_event.set()
        self.collector.join()
        self.logger.info("Video collector stopped!")
        self.color_queue.put(None)
        self.depth_queue.put(None)
        time.sleep(0.1)
        self.depth_process_stop_event.set()
        self.color_process_stop_event.set()
        self.depth_process.join(timeout=1)
        self.logger.info("Depth saver stopped!")
        self.color_process.join(timeout=1)
        self.logger.info("RGB saver stopped!")
        self.logger.info("Camera stopped!")


    def stop_read(self):
        self.collector_thread_stop_event.set()
        self.collector.join()
        self.logger.info("Video collector stopped!")
        self.logger.info("Camera read stopped!")

    def stop_write(self):
        self.color_queue.put(None)
        self.depth_queue.put(None)
        time.sleep(0.1)
        self.depth_process_stop_event.set()
        self.color_process_stop_event.set()
        self.depth_process.join(timeout=1)
        self.logger.info("Depth saver stopped!")
        self.color_process.join(timeout=1)
        self.logger.info("RGB saver stopped!")
        self.logger.info("Camera write stopped!")

    def get_observation(self):
        return self.collector.get_observation()


class ProprioceptionSaveThread(threading.Thread):
    def __init__(self, base_operator, arm_and_lift_operator, gripper_operator, save_dir, sample_rate, stop_event, logger=None):
        super().__init__()
        self.base_operator = base_operator
        self.arm_and_lift_operator = arm_and_lift_operator
        self.gripper_operator = gripper_operator
        self.save_dir = save_dir
        self.sample_rate = sample_rate
        self.logger = logger

        self.stop_event = stop_event

    def run(self):
        try:
            if self.logger is not None:
                self.logger.info("Proprio data saver started!")
            save_filename = os.path.join(self.save_dir, "robot_proprioception.txt")
            step_time = 1.0 / self.sample_rate
            with open(save_filename, "a") as wf:
                while not self.stop_event.is_set():
                    t1 = time.time()
                    base_states, base_velocity_states = self.base_operator.get_observation()
                    arm_and_lift_states = self.arm_and_lift_operator.get_observation()
                    gripper_states = self.gripper_operator.get_observation()
                    if base_states is not None and base_velocity_states is not None and \
                        arm_and_lift_states is not None and gripper_states is not None:
                        proprioception = {"base_states": base_states,
                                          "base_velocity_states": base_velocity_states,
                                          "arm_and_lift_states": arm_and_lift_states,
                                          "gripper_states": gripper_states,
                                          "timestamp": time.time() * 1000}
                        wf.write(f'{json.dumps(proprioception)}\n\n')
                        wf.flush()
                    else:
                        time.sleep(0.01)
                    t2 = time.time()
                    sleep_time = max(0.0001, step_time - (t2 - t1))
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            self.logger.warn("KeyboardInterrupt: Proprio data saver")
        except Exception as e:
            self.logger.error(f'Abnormal exit: Proprio data saver. Exception:{e}')


class RobotAixer(object):
    def __init__(self,
                 save_dir,
                 save_data=True,
                 arm_and_lift_controller_ip='192.168.10.18',
                 arm_and_lift_controller_port=8080,
                 gripper_ip='192.168.10.18',
                 gripper_port=8080,
                 base_ip='192.168.10.10',
                 base_port=31001,
                 host_ip="192.168.10.100",
                 host_port=10003,
                 logger=None
                 ):
        super().__init__()
        self.logger = logger
        self.save_data = save_data
        self.arm_and_lift_operator = ArmAndLift(arm_and_lift_controller_ip, host_ip, host_port, logger)
        self.gripper_operator = Gripper(gripper_ip, gripper_port, sample_frequency=10, logger=logger)
        self.base_operator = Base(base_ip, base_port, sample_frequency=10, logger=logger)
        self.head_camera_operator = Camera(save_dir, "head", sample_rate=30, width=640, height=480, save_data=save_data, show=1, logger=logger)
        self.wrist_camera_operator = Camera(save_dir, "wrist", sample_rate=30, width=640, height=480, save_data=save_data, show=1, logger=logger)
        if save_data:
            self.save_thread_stop_event = threading.Event()
            self.save_thread = ProprioceptionSaveThread(self.base_operator, self.arm_and_lift_operator, self.gripper_operator, save_dir, sample_rate=30, stop_event=self.save_thread_stop_event, logger=logger)

    def start(self):
        self.arm_and_lift_operator.start()
        self.gripper_operator.start()
        self.base_operator.start()
        if self.save_data:
            self.save_thread.start()
        self.head_camera_operator.start()
        self.wrist_camera_operator.start()
        self.logger.info("Robot Aixer started!")

    def stop(self):
        self.arm_and_lift_operator.stop()
        self.gripper_operator.stop()
        self.base_operator.stop()
        self.head_camera_operator.stop()
        self.wrist_camera_operator.stop()
        if self.save_data:
            self.save_thread_stop_event.set()
            self.save_thread.join()
            self.logger.info("Proprio data saver stopped!")
        self.logger.info("Robot Aixer stopped!")

    def stop_read(self):
        self.arm_and_lift_operator.stop_read()
        self.gripper_operator.stop_read()
        self.base_operator.stop_read()
        self.head_camera_operator.stop_read()
        self.wrist_camera_operator.stop_read()
        self.logger.info("Robot Aixer read stopped!")

    def stop_write(self):
        self.arm_and_lift_operator.stop_write()
        self.gripper_operator.stop_write()
        self.base_operator.stop_write()
        self.head_camera_operator.stop_write()
        self.wrist_camera_operator.stop_write()
        if self.save_data:
            self.save_thread_stop_event.set()
            self.save_thread.join()
            self.logger.info("Proprio data saver stopped!")
        self.logger.info("Robot Aixer write stopped!")

    def get_observation(self):
        return self.base_operator.get_observation(), \
               self.arm_and_lift_operator.get_observation(), \
               self.gripper_operator.get_observation(), \
               self.head_camera_operator.get_observation(), \
               self.wrist_camera_operator.get_observation()

    def get_head_camera_observation(self):
        return self.head_camera_operator.get_observation()

    def get_wrist_camera_observation(self):
        return self.wrist_camera_operator.get_observation()

    def get_base_observation(self):
        return self.base_operator.get_observation()

    def get_gripper_observation(self):
        return self.gripper_operator.get_observation()

    def get_arm_and_lift_observation(self):
        return self.arm_and_lift_operator.get_observation()

    def control(self,
                base_linear_velocity=0,
                base_angular_velocity=0,
                lift_height=None,
                arm_pose=None,
                gripper_displacement=None
                ):
        self.base_operator.control(base_linear_velocity, base_angular_velocity)
        self.arm_and_lift_operator.control(lift_height, arm_pose)
        self.gripper_operator.control(gripper_displacement)

    def init_pose(self):
        self.gripper_operator.control(0)
        self.arm_and_lift_operator.init_pose()


