import threading
import numpy as np
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from collision_detector.srv import CollisionState, CollisionStateRequest
from ros_operator_command.srv import RosCommand, RosCommandRequest

class RosOperator:
    def __init__(self, args, use_collision = True):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.puppet_arm_left_pos_deque = None
        self.puppet_arm_right_pos_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.puppet_arm_left_pos_publisher = None
        self.puppet_arm_right_pos_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.collision_client = None
        self.command_server = None
        self.args = args
        self.ctrl_state = False
        self.use_collision = use_collision
        self.ctrl_state_lock = threading.Lock()
        self.run_state = True
        self.all_time_actions = None
        self.init()
        self.init_ros()

        self.last_arm_joint_state_puppet_left_position = None
        self.last_arm_joint_state_puppet_right_position = None

        self.k = 3
        self.times = np.array([i for i in range(self.k)])
        self.arm_joint_state_puppet_left_publish_list = []
        self.arm_joint_state_puppet_right_publish_list = []

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.puppet_arm_left_pos_deque = deque()
        self.puppet_arm_right_pos_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()
        self.all_time_actions = np.zeros([
            self.args.max_publish_step, 
            self.args.max_publish_step + self.args.chunk_size, 
            self.args.state_dim])

    def puppet_arm_publish(self, left, right):
        """publish joint state
        Args:
            left : joint state of left arm
            right: joint state of right arm
        """
        if not self.run_state:
            return
        
        if self.args.use_joint:
            self.puppet_arm_joint_publish(left, right)
        else:
            self.puppet_arm_ee_publish(left, right)
    
    def puppet_arm_joint_publish(self, left, right):
        if self.use_collision:
            try:
                collision_request = CollisionStateRequest()
                collision_request.left = left
                collision_request.right = right
                collision_response = self.collision_client.call(collision_request)
                if collision_response.result :
                    print("Collision!!!")
                    return
            except rospy.ServiceException as e:
                print("Service call failed: %s", e)
        
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)
        self.last_arm_joint_state_puppet_left_position = left
        self.last_arm_joint_state_puppet_right_position = right
    
    def puppet_arm_ee_publish(self, left, right):
        if len(left) != 7 or len(right) != 7:
            return
        
        left_ee_state_msg = PoseStamped()
        left_ee_state_msg.header = Header()
        left_ee_state_msg.header.stamp = rospy.Time.now()
        left_ee_state_msg.pose.position.x = left[0]
        left_ee_state_msg.pose.position.y = left[1]
        left_ee_state_msg.pose.position.z = left[2]
        left_ee_state_msg.pose.orientation.x = left[3]
        left_ee_state_msg.pose.orientation.y = left[4]
        left_ee_state_msg.pose.orientation.z = left[5]
        left_ee_state_msg.pose.orientation.w = left[6]

        right_ee_state_msg = PoseStamped()
        right_ee_state_msg.header = Header()
        right_ee_state_msg.header.stamp = rospy.Time.now()
        right_ee_state_msg.pose.position.x = right[0]
        right_ee_state_msg.pose.position.y = right[1]
        right_ee_state_msg.pose.position.z = right[2]
        right_ee_state_msg.pose.orientation.x = right[3]
        right_ee_state_msg.pose.orientation.y = right[4]
        right_ee_state_msg.pose.orientation.z = right[5]
        right_ee_state_msg.pose.orientation.w = right[6]
        self.puppet_arm_left_pos_publisher.publish(left_ee_state_msg)
        self.puppet_arm_right_pos_publisher.publish(right_ee_state_msg)

    def robot_base_publish(self, vel):
        """ publish robot chassis status

        Args:
            vel : linear.x and angular.z
        """
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and left_arm is None and right_arm is None and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            # self.puppet_arm_publish(left_arm, right_arm)
            self.puppet_arm_joint_publish(left_arm, right_arm)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()
        
    def puppet_arm_pos_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and left_arm is None and right_arm is None and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_pos_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_pos_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            # self.puppet_arm_publish(left_arm, right_arm)
            self.puppet_arm_ee_publish(left_arm, right_arm)
            step += 1
            print("puppet_arm_pos_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def arm_joint_state_puppet_publish_interpolation_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.arm_joint_state_puppet_publish_interpolation, args=(left, right))
        self.puppet_arm_publish_thread.start()
        self.puppet_arm_publish_thread.join()
    
    def arm_joint_state_puppet_publish_interpolation(self, left, right):
        # print("left_target:", left)
        # print("right_target:", right)
        arm_left = self.last_arm_joint_state_puppet_left_position
        arm_right = self.last_arm_joint_state_puppet_right_position
        rate = rospy.Rate(200)
        while True and arm_left is None and arm_right is None and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            
            if len(self.puppet_arm_left_deque) != 0:
                arm_left = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                arm_right = list(self.puppet_arm_right_deque[-1].position)
            if arm_left is None or arm_right is None:
                rate.sleep()
                continue
            else:
                break
      
        arm = np.concatenate((np.array(arm_left), np.array(arm_right)), axis=0)
        arm_target = np.concatenate((np.array(left), np.array(right)), axis=0)
        
        arm_diff = arm_target - arm
        if max(arm_diff) > 0.5:
            rate = rospy.Rate(30)
            left_symbol = [1 if left[i] - arm_left[i] > 0 else -1 for i in range(len(left))]
            right_symbol = [1 if right[i] - arm_right[i] > 0 else -1 for i in range(len(right))]
            flag = True
            step = 0
            while flag and not rospy.is_shutdown():
                if self.puppet_arm_publish_lock.acquire(False):
                    return
                left_diff = [abs(left[i] - arm_left[i]) for i in range(len(left))]
                right_diff = [abs(right[i] - arm_right[i]) for i in range(len(right))]
                flag = False
                for i in range(len(left)):
                    if left_diff[i] < self.args.arm_steps_length[i]:
                        arm_left[i] = left[i]
                    else:
                        arm_left[i] += left_symbol[i] * self.args.arm_steps_length[i]
                        flag = True
                for i in range(len(right)):
                    if right_diff[i] < self.args.arm_steps_length[i]:
                        arm_right[i] = right[i]
                    else:
                        arm_right[i] += right_symbol[i] * self.args.arm_steps_length[i]
                        flag = True
                self.puppet_arm_publish(arm_left, arm_right)
                step += 1
                rate.sleep()
        else:
            if len(self.arm_joint_state_puppet_left_publish_list) == 0 or len(self.arm_joint_state_puppet_right_publish_list) == 0:
                for i in range(self.k):
                    self.arm_joint_state_puppet_left_publish_list.append(arm_left)
                    self.arm_joint_state_puppet_right_publish_list.append(arm_right)
            self.arm_joint_state_puppet_left_publish_list.append(left)
            self.arm_joint_state_puppet_right_publish_list.append(right)
            self.arm_joint_state_puppet_left_publish_list = self.arm_joint_state_puppet_left_publish_list[-self.k:]
            self.arm_joint_state_puppet_right_publish_list = self.arm_joint_state_puppet_right_publish_list[-self.k:]
            left_positions = [[self.arm_joint_state_puppet_left_publish_list[k][i] for k in range(self.k)] for i in range(len(left))]
            right_positions = [[self.arm_joint_state_puppet_right_publish_list[k][i] for k in range(self.k)] for i in range(len(right))]
            
            left_coeffs = [self.interpolation_param(left_positions[i]) for i in range(len(left_positions))]
            right_coeffs = [self.interpolation_param(right_positions[i]) for i in range(len(right_positions))]

            hz = 200
            step = 10
            rate = rospy.Rate(hz)
            for i in range(step):
                if self.puppet_arm_publish_lock.acquire(False):
                    return
                left_arm = [np.polyval(left_coeffs[j][::-1], (self.k - 2) + (i + 1) * 0.05) for j in range(len(left_coeffs))]
                right_arm = [np.polyval(right_coeffs[j][::-1], (self.k - 2) + (i + 1) * 0.05) for j in range(len(right_coeffs))]
                self.puppet_arm_publish(left_arm, right_arm)
                rate.sleep()

    def interpolation_param(self, positions):
        positions = np.array(positions)
        # 构建矩阵A和向量b
        A = [np.ones_like(self.times)]
        for i in range(self.k - 1):
            A.append(self.times ** (i + 1))
        A = np.vstack(A).T
        b = positions
        # 解线性方程组得到多项式系数
        coeffs = np.linalg.solve(A, b)
        # 使用多项式系数计算给定时间的速度
        return coeffs

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if not self.run_state:
            return False
        
        if self.args.use_joint:
            return self.get_joint_frame()
        else:
            return self.get_ee_frame()
        
    def get_client_frame(self):
        obs = {}
        while True:
            if self.args.use_joint:
                result = self.get_joint_frame()
                if result == False:
                    continue
                (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                    puppet_arm_left, puppet_arm_right, robot_base) = result
                obs["image_primary"] = img_front
                obs["image_left_wrist"] = img_left
                obs["image_right_wrist"] = img_right
                obs["proprio"] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
                break            
            else:
                result = self.get_ee_frame()
                if result == False:
                    continue
                img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,\
                    puppet_arm_left_pos, puppet_arm_right_pos, robot_base = result
                obs["image_primary"] = img_front
                obs["image_left_wrist"] = img_left
                obs["image_right_wrist"] = img_right
                obs["proprio"] = np.concatenate((np.array(puppet_arm_left_pos.position), np.array(puppet_arm_right_pos.position)), axis=0)
                break
            
        return obs
        
    def get_joint_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def get_ee_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_pos_deque) == 0 or self.puppet_arm_left_pos_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_pos_deque) == 0 or self.puppet_arm_right_pos_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_pos_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_pos_deque.popleft()
        puppet_arm_left_pos = self.puppet_arm_left_pos_deque.popleft()

        while self.puppet_arm_right_pos_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_pos_deque.popleft()
        puppet_arm_right_pos = self.puppet_arm_right_pos_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left_pos, puppet_arm_right_pos, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def puppet_arm_left_pos_callback(self, msg):
        if len(self.puppet_arm_left_pos_deque) >= 2000:
            self.puppet_arm_left_pos_deque.popleft()
        self.puppet_arm_left_pos_deque.append(msg)

    def puppet_arm_right_pos_callback(self, msg):
        if len(self.puppet_arm_right_pos_deque) >= 2000:
            self.puppet_arm_right_pos_deque.popleft()
        self.puppet_arm_right_pos_deque.append(msg)
             
    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state
    
    def resetting(self):
        if self.args.use_joint:
            # 发布基础的姿态
            left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
            right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
            left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
            right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
            
            self.puppet_arm_publish_continuous(left0, right0)
            self.puppet_arm_publish_continuous(left1, right1)
            # self.puppet_arm_publish_linear(left0, right0)
            # self.puppet_arm_publish_linear(left1, right1)
        else :
            left0 = [-0.003645, -0.001574671, -0.012768886, -0.09131508, 0.25385445, -0.025179114, 3.557830810546875]
            right0 = [-0.005166925, -0.007107213, -0.011377825, 0.030795027, 0.2241224, -0.14790899, 3.557830810546875]
            left1 = [-0.003645, -0.001574671, -0.012768886, -0.09131508, 0.25385445, -0.025179114, -0.3393220901489258]
            right1 = [-0.005166925, -0.007107213, -0.011377825, 0.030795027, 0.2241224, -0.14790899, -0.3397035598754883]

            self.puppet_arm_pos_publish_continuous(left0, right0)
            self.puppet_arm_pos_publish_continuous(left1, right1)

        self.clear_deque()

    def clear_deque(self): 
        self.img_front_deque.clear()
        self.img_left_deque.clear()
        self.img_right_deque.clear()
        self.img_front_depth_deque.clear()
        self.img_left_depth_deque.clear()
        self.img_right_depth_deque.clear()
        self.puppet_arm_left_deque.clear()
        self.puppet_arm_right_deque.clear()

        self.all_time_actions = np.zeros([
            self.args.max_publish_step, 
            self.args.max_publish_step + self.args.chunk_size, 
            self.args.state_dim])
    
    def command_operator(self, req):
        if req.command == 'pause':
            self.run_state = False
        elif req.command == 'start':
            self.run_state = True
        elif req.command == 'reset':
            self.resetting()
        else:
            rospy.logwarn('Invalid command %s' % req.command)
            return False
        
        return True

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_pos, PoseStamped, self.puppet_arm_left_pos_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_pos, PoseStamped, self.puppet_arm_right_pos_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_left_pos_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_pos, PoseStamped, queue_size=10)
        self.puppet_arm_right_pos_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_pos, PoseStamped, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)

        self.command_server = rospy.Service("ros_operator/command", RosCommand, self.command_operator)

        if self.use_collision:
            print("Wait for collision service.")
            rospy.wait_for_service('/collision_state')
            print("connected server")
            self.collision_client = rospy.ServiceProxy('/collision_state', CollisionState)