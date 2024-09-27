#--coding:utf-8--
import os
import time
import pdb
import argparse
import logging
from inputs import get_gamepad
from ainnorobot_data.collector_realman.joystick import Rocker, ArmPoseController
from ainnorobot_data.collector_realman.robot_aixer import RobotAixer


def create_logger(log_dir):
    logger = logging.Logger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.txt'), mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="control robot and collect states data")
    parser.add_argument("--operator", type=str, choices=["jmt", "gdh", "ljh"], default="jmt",
                        help="who is conducting the experiment.")
    parser.add_argument("--scene", type=str, default="工业",
                        help="the scene of experiment. default is `工业`.")
    parser.add_argument("--environment", type=str, default="创新奇智研发实验室",
                        help="the experiment environment. default is `创新奇智研发实验室`.")
    parser.add_argument("--sample_rate", type=int, default=30,
                        help="the data sample rate. default is 30.")
    parser.add_argument("--save_dir", type=str, default="/home/rm/CaptureData", help="data save directory.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.sample_rate > 0, "sample rate must be positive integer!"

    t_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
    save_dir = os.path.join(args.save_dir, f"{t_str}")
    os.makedirs(save_dir, exist_ok=True)

    logger = create_logger(save_dir)

    '''---------------------------选择任务--------------------------'''

    logger.info('=====START=====')
    logger.info(f'operator:{args.operator}')
    logger.info(f'scene:{args.scene}')
    logger.info(f'environment:{args.environment}')
    logger.info(f'sample_rate:{args.sample_rate}')

    task_name = {'1': "整理一下工作台",
                 '2': "把笔放到笔筒里",
                 '3': "把手套放到框里",
                 '4': "把螺丝刀放到抽屉里",
                 '5': "把螺丝刀放到框里",
                 '6': "把水杯放到笔筒旁边",
                 '7': "把水杯放到桌子右上角",
                 '8': "把空瓶扔到垃圾桶里",
                 '9': '擦一下工作台',
                 '10': "放下东西"
                 }
    for key in task_name:
        print(f'{key}:{task_name[key]}')
    time.sleep(0.1)

    curr_task = None
    while True:
        task = input("请选择任务:")
        if task in task_name:
            print(f"选择任务{task}:{task_name[task]}")
            logger.info(f'task_name:{task_name[task]}')
            curr_task = task_name[task]
            break
        else:
            print("没有此任务,请重新选择")

    '''---------------------------创建采集器--------------------------'''
    aixer = RobotAixer(save_dir, save_data=True, logger=logger)
    audio_filepath = os.path.join(save_dir, "audio.txt")

    # 手柄相关初始化
    arm_pose_controller = ArmPoseController(aixer.arm_and_lift_operator.operator)
    arm_pose_controller.start()
    l_rocker = Rocker(aixer, name="left")
    r_rocker = Rocker(aixer, name="right")

    isRun = False
    isOpen = True
    complete = False

    try:
        while not complete:
            try:
                events = get_gamepad()
            except Exception as e:
                logger.error("game pad not found!")
                exit(-1)
            for event in events:
                code = event.code
                state = event.state
                if 'BTN_START' == code and state == 1:
                    if not isRun:
                        logger.info(f'BTN_START:{time.time() * 1000}')
                        isRun = True
                        aixer.start()
                        # 模拟的语音数据
                        with open(audio_filepath, 'a', encoding='utf-8') as file:
                            file.write(f'{time.time() * 1000},{curr_task}\n')
                            file.flush()
                            file.close()
                if 'BTN_SELECT' == code and state == 1:
                    logger.info(f'BTN_SELECT:{time.time() * 1000}')
                    if isRun:
                        aixer.init_pose()
                        aixer.stop_read()
                        time.sleep(1)
                        complete = True
                        isRun = False
                        break
                    else:
                        aixer.init_pose()
                if not isRun:
                    time.sleep(0.01)
                    continue
                if 'ABS_HAT0X' == code and state == -1:
                    # 左按钮
                    arm_pose_controller.set_y(1)
                if 'ABS_HAT0X' == code and state == 1:
                    # 右按钮
                    arm_pose_controller.set_y(-1)
                if 'ABS_HAT0X' == code and state == 0:
                    arm_pose_controller.set_y(0)
                if 'ABS_HAT0Y' == code and state == -1:
                    # 上按钮
                    arm_pose_controller.set_x(1)
                if 'ABS_HAT0Y' == code and state == 1:
                    # 下按钮
                    arm_pose_controller.set_x(-1)
                if 'ABS_HAT0Y' == code and state == 0:
                    arm_pose_controller.set_x(0)
                if 'BTN_SOUTH' == code and state == 1:
                    # A按钮
                    arm_pose_controller.set_rx(-1)
                if 'BTN_SOUTH' == code and state == 0:
                    arm_pose_controller.set_rx(0)
                if 'BTN_EAST' == code and state == 1:
                    # B按钮
                    arm_pose_controller.set_ry(-1)
                if 'BTN_EAST' == code and state == 0:
                    arm_pose_controller.set_ry(0)
                if 'BTN_NORTH' == code and state == 1:
                    # X按钮
                    arm_pose_controller.set_ry(1)
                if 'BTN_NORTH' == code and state == 0:
                    arm_pose_controller.set_ry(0)
                if 'BTN_WEST' == code and state == 1:
                    # Y按钮
                    arm_pose_controller.set_rx(1)
                if 'BTN_WEST' == code and state == 0:
                    arm_pose_controller.set_rx(0)
                if 'BTN_TL' == code and state == 1:
                    arm_pose_controller.set_z(1)
                if 'BTN_TL' == code and state == 0:
                    arm_pose_controller.set_z(0)
                if 'BTN_TL2' == code and state == 1:
                    arm_pose_controller.set_z(-1)
                if 'BTN_TL2' == code and state == 0:
                    arm_pose_controller.set_z(0)
                if 'BTN_TR' == code and state == 1:
                    arm_pose_controller.set_rz(1)
                if 'BTN_TR' == code and state == 0:
                    arm_pose_controller.set_rz(0)
                if 'BTN_TR2' == code and state == 1:
                    arm_pose_controller.set_rz(-1)
                if 'BTN_TR2' == code and state == 0:
                    arm_pose_controller.set_rz(0)
                if 'ABS_X' == code:
                    l_rocker.set_x(state)
                if 'ABS_Y' == code:
                    l_rocker.set_y(state)
                if 'ABS_Z' == code:
                    r_rocker.set_x(state)
                if 'ABS_RZ' == code:
                    r_rocker.set_y(state)
                if 'BTN_THUMBR' == code and state == 0:
                    if isOpen:
                        aixer.gripper_operator.control(displacement=255)
                        isOpen = False
                    else:
                        aixer.gripper_operator.control(displacement=0)
                        isOpen = True

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: exit!")
        aixer.arm_and_lift_operator.operator.Set_Realtime_Push(enable=False)
        aixer.arm_and_lift_operator.operator.RM_API_UnInit()
        aixer.arm_and_lift_operator.operator.Arm_Socket_Close()
    except Exception as e:
        logger.info(f"Abnormal exit. Exception: {e}")
    finally:
        logger.info(f'finally:{time.time() * 1000}')
        arm_pose_controller.stop()
        l_rocker.stop_send_velocity()
        r_rocker.stop_send_velocity()
        aixer.stop_write()
        aixer.arm_and_lift_operator.operator.Set_Realtime_Push(enable=False)
        aixer.arm_and_lift_operator.operator.RM_API_UnInit()
        aixer.arm_and_lift_operator.operator.Arm_Socket_Close()
        logger.info('=====END=====')

