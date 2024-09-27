import logging
import os
from flask import Flask, Response, render_template  
import cv2  
import threading  
import time

import numpy as np  
  
class VideoRenderer:  
    def __init__(self, host='0.0.0.0', port=5000, resize=None): 
        template_folder=os.path.join(os.path.dirname(__file__),"templates")
        self.app = Flask(__name__,template_folder=template_folder) 
        self.app.logger.setLevel(logging.ERROR) 
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.host = host  
        self.port = port  
        self.lock = threading.Lock()  
        self.frame = None  
        self.running = False
        self.resize = resize  
  
    def update_frame(self, frame:np.array):  
        with self.lock:  
            self.frame = frame  
  
    def get_frame(self):  
        with self.lock:  
            if self.frame is None:  
                return None  
            frame = self.frame
            #这里取完数据立即释放锁，耗时的处理逻辑要放到加锁范围之外
        
        frame = frame[:,:,::-1]  
        if self.resize is not None:
            frame = cv2.resize(frame,self.resize)
        ret, buffer = cv2.imencode('.jpg', frame)  
        frame = buffer.tobytes()  
        return frame  
  
    def start(self):  
        def generate():  
            while self.running:  
                frame = self.get_frame()  
                if frame is not None:  
                    yield (b'--frame\r\n'  
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
                else:  
                    time.sleep(0.1)  
  
        @self.app.route('/video_feed')  
        def video_feed():  
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/')  
        def index():  
            return render_template('index.html')   
  
        self.running = True 

        def _start():
            self.app.run(host=self.host, port=self.port, threaded=True)
        threading.Thread(target=_start).start()
        
        print(f"server started")  
  
    def stop(self):  
        self.running = False  
  
# 使用示例  
if __name__ == '__main__':  
    renderer = VideoRenderer()  
    cap = cv2.VideoCapture(0)  # 从摄像头读取视频  
    try:  
        renderer.start()  
        while True:  
            ret, frame = cap.read()  
            if not ret:  
                break  
            renderer.update_frame(frame)  
    finally:  
        renderer.stop()  
        cap.release()