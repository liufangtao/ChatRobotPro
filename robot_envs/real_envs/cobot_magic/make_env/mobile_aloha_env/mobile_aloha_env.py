import gym
import argparse
from ros_operator import RosOperator

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--use_joint', action='store', type=str, help='use_joint',
                        default=True, required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_pos', action='store', type=str, help='puppet_arm_left_cmd_pos',
                        default='master/end_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_pos', action='store', type=str, help='puppet_arm_right_cmd_pos',
                        default='master/end_right', required=False) 
    parser.add_argument('--puppet_arm_left_pos', action='store', type=str, help='puppet_arm_left_pos',
                        default='/puppet/end_left', required=False)
    parser.add_argument('--puppet_arm_right_pos', action='store', type=str, help='puppet_arm_right_pos',
                        default='/puppet/end_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for gym
    parser.add_argument('--task', action='store', type=str, help='tast', default='', required=False)
    # parser.add_argument('--task', action='store', type=str, help='tast', default='', required=False)

    args = parser.parse_args()
    return args

class MobileAlohaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.task = ''
        self.ros_operator = RosOperator(get_arguments(), False)

    def reset(self):
        self.ros_operator.resetting()

        return self.ros_operator.get_client_frame(), None

    def step(self, action):
        left_action = action[:7]  # 取7维度
        right_action = action[7:14]
        # self.ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
        self.ros_operator.arm_joint_state_puppet_publish_interpolation_thread(left_action, right_action)  # puppet_arm_publish_continuous_thread
        # if self.args.use_robot_base:
        #     vel_action = action[14:16]
        #     self.ros_operator.robot_base_publish(vel_action)

        return self.ros_operator.get_client_frame(), 0, False, False, None


    



