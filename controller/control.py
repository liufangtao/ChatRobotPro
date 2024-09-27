import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7' 

import argparse
import sys  
import time
import queue
import threading
from threading import Timer

import gym

from remote_policy import RemotePolicy
 
# sys.path.append("/data/maohui/robotics/act-plus-plus")
sys.path.append("./")
sys.path.append("./robot_envs/real_envs/cobot_magic/make_env/")
sys.path.append("./robot_envs/real_envs/cobot_magic/make_env/mobile_aloha_env/")
import mobile_aloha_env

class RepeatTimer(Timer):
    def __init__(self, interval, env, policy):
        super().__init__(interval, self.remote_policy)
        self.obs = {}
        self.env = env
        self.policy = policy
        self.t = time.time()

    def run(self):
        # while not self.finished.wait(self.interval):
        while True:
            self.function(*self.args, **self.kwargs)
    
    def start_inference(self):
        self.obs, _ = self.env.reset()
        self.start()

    def remote_policy(self):
        start_time = time.time()
        actions = self.policy(self.obs, "Grap the coke with your left arm and put it into the plate")
        end_time = time.time()
        print('remote policy model cost time: ', end_time - start_time)
        actions = actions[0]

        c = end_time - self.t
        start = time.time()
        self.obs, reward, done, trunc, info = self.env.step(actions)   
        end = time.time()
        print('step cost time: ', end - start)
        time.sleep(max(0, 0.1-c))
        self.t = time.time()

def get_arguments():
    parser = argparse.ArgumentParser()

    # for gym
    parser.add_argument('--task', action='store', type=str, help='tast', default='', required=False)

    parser.add_argument('--ip', type=str, default="192.168.10.82", help='IP address of the gRPC server')  
    # parser.add_argument('--ip', type=str, default="127.0.0.1", help='IP address of the gRPC server')  
    parser.add_argument('--port', type=int, default=50051, help='Port number of the gRPC server')  

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    env = gym.make("mobile_aloha_env/MobileAloha-v0")

    # 连接到gRPC服务器  
    policy = RemotePolicy(args.ip, args.port)

    timer = RepeatTimer(0.1, env, policy)
    timer.start_inference()

  
if __name__ == '__main__':  
    main()
  