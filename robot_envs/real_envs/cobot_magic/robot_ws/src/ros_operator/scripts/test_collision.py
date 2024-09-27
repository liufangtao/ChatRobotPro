import argparse
import numpy as np
import rospy
from collision_detector.srv import CollisionState, CollisionStateRequest
from ros_operator import RosOperator

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

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
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
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

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

    # for gym
    parser.add_argument('--task', action='store', type=str, help='tast', default='', required=False)
    # parser.add_argument('--task', action='store', type=str, help='tast', default='', required=False)

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    ros_operator = RosOperator(args)

    while True and not rospy.is_shutdown():
        if len(ros_operator.puppet_arm_left_deque) == 0 or len(ros_operator.puppet_arm_right_deque) == 0:
            continue
        frame_time = min([ros_operator.puppet_arm_left_deque[-1].header.stamp.to_sec(),
                           ros_operator.puppet_arm_right_deque[-1].header.stamp.to_sec()])

        if ros_operator.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            continue
        if ros_operator.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            continue
        
        while ros_operator.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            ros_operator.puppet_arm_left_deque.popleft()
        puppet_arm_left = ros_operator.puppet_arm_left_deque.popleft()

        while ros_operator.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            ros_operator.puppet_arm_right_deque.popleft()
        puppet_arm_right = ros_operator.puppet_arm_right_deque.popleft()

        left = np.array(puppet_arm_left.position)
        right = np.array(puppet_arm_right.position)

        try:
            collision_request = CollisionStateRequest()
            collision_request.left = left
            collision_request.right = right
            collision_response = ros_operator.collision_client.call(collision_request)
            if collision_response.result :
                print("Collision!!!")
                continue
            else:
                print("normal")
        except rospy.ServiceException as e:
            print("Service call failed: %s", e)
    

if __name__ == '__main__':
    main()