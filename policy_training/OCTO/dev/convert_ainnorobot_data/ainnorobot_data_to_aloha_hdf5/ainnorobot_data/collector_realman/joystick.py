import copy
import time
import threading
import numpy as np


def left_rocker_callback(x, y, robot=None):
    if x == 0 and y == 0:
        robot.base_operator.control(0, 0)
    else:
        if x > 0:
            robot.base_operator.control(0, 0.2)  # linear_velocity angular_velocity
        if x < 0:
            robot.base_operator.control(0, -0.2)
        if y > 0:
            robot.base_operator.control(0.05, 0)
        if y < 0:
            robot.base_operator.control(-0.05, 0)


def right_rocker_callback(x, y, robot=None):
    if y == 0:
        robot.arm_and_lift_operator.control_lift(0)
    else:
        if y < 0:
            robot.arm_and_lift_operator.control_lift(-30)
        if y > 0:
            robot.arm_and_lift_operator.control_lift(30)


class Rocker:

    def __init__(self, robot, name="left"):
        self.name = name
        self.robot = robot
        self.x = 128
        self.y = 128
        self.out_x = 0
        self.out_y = 0
        if name == "left":
            self.callback = left_rocker_callback
            self.send = SendVelocity(self.callback, robot)
            self.send.start()
        elif name == "right":
            self.callback = right_rocker_callback
        else:
            raise ValueError

    def set_x(self, x):
        self.x = x
        self.speed()

    def set_y(self, y):
        self.y = y
        self.speed()

    def speed(self):
        if self.x == 128 and self.y == 128:
            if self.name == "left":
                self.send.stop_send()
            self.out_x = 0
            self.out_y = 0
            self.callback(0, 0, self.robot)
        else:
            if self.out_x == 0 and self.out_y == 0:
                self.out_x = 128 - self.x
                self.out_y = 128 - self.y
                self.out_x = 1 if self.out_x > 0 else (-1 if self.out_x < 0 else 0)
                self.out_y = 1 if self.out_y > 0 else (-1 if self.out_y < 0 else 0)
                if self.name == "left":
                    self.send.start_send(self.out_x, self.out_y)
                else:
                    self.callback(self.out_x, self.out_y, self.robot)

    def stop_send_velocity(self):
        if self.name == "left":
            self.send.stop()


class SendVelocity(threading.Thread):

    def __init__(self, call_back, robot=None):
        super().__init__()
        self.robot = robot
        self.x = 0
        self.y = 0
        self.send = False
        self.callback = call_back
        self.stop_event = threading.Event()

    def run(self):
        while True:
            if self.stop_event.is_set():
                break
            if self.send:
                self.callback(self.x, self.y, self.robot)
                time.sleep(0.3)
            else:
                time.sleep(0.3)

    def stop(self):
        self.stop_event.set()

    def start_send(self, x, y):
        self.x = x
        self.y = y
        self.send = True

    def stop_send(self):
        self.send = False


class ArmPoseController(threading.Thread):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.x = 0
        self.y = 0
        self.z = 0
        self.rx = 0
        self.ry = 0
        self.rz = 0
        self.joints = []
        self.pose_x = 0
        self.pose_y = 0
        self.pose_z = 0
        self.pose_rx = 0
        self.pose_ry = 0
        self.pose_rz = 0
        self.stop_event = threading.Event()
        # self.init_current()

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    def set_rx(self, rx):
        self.rx = rx

    def set_ry(self, ry):
        self.ry = ry

    def set_rz(self, rz):
        self.rz = rz

    def init_current(self):
        code, joints, pose, _, _ = self.arm.Get_Current_Arm_State()
        self.joints = joints
        self.pose_x = round(pose[0], 6)
        self.pose_y = round(pose[1], 6)
        self.pose_z = round(pose[2], 6)
        self.pose_rx = round(pose[3], 3)
        self.pose_ry = round(pose[4], 3)
        self.pose_rz = round(pose[5], 3)

    def run(self):
        sleep_time = 0.01
        while True:
            if self.stop_event.is_set():
                break
            if self.x == 0 and self.y == 0 and self.z == 0 and self.rx == 0 and self.ry == 0 and self.rz == 0:
                self.init_current()
                time.sleep(0.06)
            else:
                # t1 = time.time()

                # # joint
                # self.init_current()
                # if self.x != 0:
                #     self.pose_x += self.x * 0.01
                # if self.y != 0:
                #     self.pose_y += self.y * 0.01
                # if self.z != 0:
                #     self.pose_z += self.z * 0.01
                # if self.rx != 0:
                #     self.pose_rx += self.rx * 0.1
                # if self.ry != 0:
                #     self.pose_ry += self.ry * 0.1
                # if self.rz != 0:
                #     self.pose_rz += self.rz * 0.1

                # eef
                if self.x != 0:
                    self.pose_x += self.x * 0.001
                if self.y != 0:
                    self.pose_y += self.y * 0.001
                if self.z != 0:
                    self.pose_z += self.z * 0.001
                if self.rx != 0:
                    self.pose_rx += self.rx * 0.015
                if self.ry != 0:
                    self.pose_ry += self.ry * 0.015
                if self.rz != 0:
                    self.pose_rz += self.rz * 0.015

                pose = [round(self.pose_x, 6),
                        round(self.pose_y, 6),
                        round(self.pose_z, 6),
                        round(self.pose_rx, 3),
                        round(self.pose_ry, 3),
                        round(self.pose_rz, 3)]
                ret = self.arm.Movep_CANFD(pose, False)

                # ret = self.arm.Algo_Inverse_Kinematics(self.joints, pose, 1)
                # target_joints = ret[1][:6]
                # self.arm.Movej_CANFD(target_joints, False, 0)

                # t2 = time.time()
                # sleep_time = max(0.001, 0.05 - (t2 - t1))
                # time.sleep(sleep_time)
                time.sleep(0.02)

    def stop(self):
        self.stop_event.set()


# class ArmPoseReader(threading.Thread):
#
#     def __init__(self, arm, frequency, stop_event, logger=None):
#         super().__init__()
#         self.arm = arm
#         self.joints = None
#         self.pose = None
#         self.frequency = frequency
#         self.logger = logger
#         self.updated = False
#         self.stop_event = stop_event
#
#     def run(self):
#         try:
#             time_step = 1.0 / self.frequency
#             if self.logger is not None:
#                 self.logger.info("Arm pose reader started!")
#             while not self.stop_event.is_set():
#                 t1 = time.time()
#                 code, joints, pose, _, _ = self.arm.Get_Current_Arm_State()
#                 self.joints = joints
#                 if pose is not None:
#                     self.pose = pose
#                     self.updated = True
#                 t2 = time.time()
#                 time_sleep = max(0.0001, time_step - (t2-t1))
#                 time.sleep(time_sleep)
#         except Exception as e:
#             self.logger.error(f'更新机械臂状态异常退出！Exception: {e}')
#
#     def get_update(self):
#         return self.joints, self.pose
#
#
#
# class ArmPoseController(threading.Thread):
#
#     def __init__(self, arm, arm_pose_reader, frequency, stop_event, logger=None):
#         super().__init__()
#         self.arm = arm
#         self.arm_pose_reader = arm_pose_reader
#         self.action = None
#         self.frequency = frequency
#         self.logger = logger
#         self.stop_event = stop_event
#
#     def run(self):
#         try:
#             time_step = 1.0 / self.frequency
#             if self.logger is not None:
#                 self.logger.info("Arm pose controller started!")
#             _, curr_pose = self.arm_pose_reader.get_update()
#             while not self.stop_event.is_set():
#                 t1 = time.time()
#                 if self.action is not None and sum(self.action) != 0:
#                     if self.arm_pose_reader.updated:
#                         _, curr_pose = self.arm_pose_reader.get_update()
#                         self.arm_pose_reader.updated = False
#                         print(f"current pose: {curr_pose}")
#                     if curr_pose is not None:
#                         target_pose = np.array(curr_pose) + np.array(self.action) * np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])
#                         ret = self.arm.Movep_CANFD(target_pose, False)
#                         print(f"target pose: {target_pose}")
#                 t2 = time.time()
#                 time_sleep = max(0.001, time_step - (t2-t1))
#                 time.sleep(time_sleep)
#                 # time.sleep(0.1)
#         except Exception as e:
#             self.logger.error(f'机械臂控制异常退出！Exception: {e}')
#
#     def set_action(self, action):
#         print(f"********************************************* set action: {action} ***********************************")
#         if self.action is None:
#             self.action = copy.deepcopy(action)
#         else:
#             self.action += copy.deepcopy(action)
#         if self.arm_pose_reader.updated:
#             self.action = [0, 0, 0, 0, 0, 0]
#         print(f"********************************************* curr action: {self.action} ***********************************")
#
#
# class ArmPoseOperator:
#     def __init__(self, arm, logger=None):
#         super().__init__()
#         self.arm = arm
#         self.logger = logger
#         self.action = [0,0,0,0,0,0]
#         self.reader_thread_stop_event = threading.Event()
#         self.reader_thread = ArmPoseReader(arm, 1, self.reader_thread_stop_event, logger)
#         self.controller_thread_stop_event = threading.Event()
#         self.controller_thread = ArmPoseController(arm, self.reader_thread, 30, self.controller_thread_stop_event, logger)
#
#     def set_x(self, x):
#         self.action[0] = x
#         self.controller_thread.set_action(self.action)
#
#     def set_y(self, y):
#         self.action[1] = y
#         self.controller_thread.set_action(self.action)
#
#     def set_z(self, z):
#         self.action[2] = z
#         self.controller_thread.set_action(self.action)
#
#     def set_rx(self, rx):
#         self.action[3] = rx
#         self.controller_thread.set_action(self.action)
#
#     def set_ry(self, ry):
#         self.action[4] = ry
#         self.controller_thread.set_action(self.action)
#
#     def set_rz(self, rz):
#         self.action[5] = rz
#         self.controller_thread.set_action(self.action)
#
#     def start(self):
#         self.reader_thread.start()
#         self.controller_thread.start()
#
#     def stop(self):
#         self.reader_thread_stop_event.set()
#         self.reader_thread.join()
#         self.logger.info("Arm pose reader stopped!")
#         self.controller_thread_stop_event.set()
#         self.controller_thread.join()
#         self.logger.info("Arm pose controller stopped!")

