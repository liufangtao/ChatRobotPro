import os
import threading
import time
from video_renderer import VideoRenderer  
import cv2  
  
renderer = VideoRenderer(port=58052)  
 
#threading.Thread(target=renderer.start).start()

print(f"cwd={os.getcwd()}")

renderer.start()  
while True:
    cap = cv2.VideoCapture('rollout_video_9_c3f98167d452e15bafe1.mp4') 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 想要的播放速率（例如，0.5为慢放一半，2为快放两倍）  
    playback_speed = 1.0  
    
    # 获取视频的帧率  
    fps = cap.get(cv2.CAP_PROP_FPS)  
    
    # 计算每帧之间的延迟时间（秒）  
    delay = (1 / (fps * playback_speed)) 

    try:  
        while True:  
            start = time.time()
            ret, frame = cap.read()  
            if not ret:  
                break  
            renderer.update_frame(frame)  
            cost = time.time()-start
            time.sleep(max(delay-cost,0))
    finally:  
        cap.release()

renderer.stop()  