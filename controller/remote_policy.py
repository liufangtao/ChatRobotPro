
import cv2
import numpy as np

import grpc
import vla_server_pb2  
import vla_server_pb2_grpc 

import sys
sys.path.append('..')

import image_process 

class RemotePolicy():
    def __init__(self, server_ip, server_port):
        channel= grpc.insecure_channel(f'{server_ip}:{server_port}')
        stub = vla_server_pb2_grpc.RobotServiceStub(channel) 
        
        # 创建Robot实例的请求  
        create_request = vla_server_pb2.CreateRobotRequest(  
            robot_type=vla_server_pb2.ALOHA_TLR,  
            description="Example robot",  
            keep_alive=3600  
        )  
        create_response = stub.CreateRobot(create_request)  
        print(f"Created robot with ID: {create_response.robot_id}") 
        self._robot_id = create_response.robot_id
        self._stub = stub

    def __call__(self, observation, language_instruction=None, unnorm_key=None):
        images = []
        for image_name in ["primary", "left_wrist", "right_wrist"]:

            image_name = "image_" + image_name
            
            image = cv2.resize(observation[image_name], (256, 256))

            # 加载JPEG图像数据  
            image_data = image_process.rgb_array_to_jpeg_bytes(np.array(image) ) 
        
            # 创建ImageData对象  
            image_message = vla_server_pb2.ImageData(  
                name=image_name,  
                encode='jpeg',  
                data=image_data  
            )  

            images.append(image_message)
        
        proprioception=[]
        if "proprio" in observation:
            state = vla_server_pb2.ProprioceptionData(
                name="proprio",
                encoding=vla_server_pb2.StateEncoding.JOINT_BIMANUAL,
                data=observation["proprio"]
            )
            proprioception.append(state)

        # 创建Observation对象  
        observation_message = vla_server_pb2.Observation(images=images, proprioception=proprioception)  
        
        task = vla_server_pb2.TaskInfo(language_instruction=language_instruction)

        # 创建RobotRequest对象  
        robot_request = vla_server_pb2.RobotRequest(  
            robot_id=self._robot_id,  
            observation=observation_message,
            task=task 
        )  
        response = self._stub.ProcessRobotRequest(robot_request) 
        if response.error_code == 0: 
            return np.array([list(a.action) for a in response.result.actions])
        else:
            print("response={response}")